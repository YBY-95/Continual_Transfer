from __future__ import print_function
import torch
import torch.nn as nn
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import os
import torch.nn.functional as F
from scipy.stats import entropy
from Transfer_loss_funcs.transfer_losses import TransferLoss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class NormalNN(nn.Module):
    '''
    Normal Neural Network with SGD for classification
    '''
    def __init__(self, agent_config):
        '''
        :param agent_config (dict): lr=float,momentum=float,weight_decay=float,
                                    schedule=[int],  # The last number in the list is the end of epoch
                                    model_type=str,model_name=str,out_dim={task:dim},model_weights=str
                                    force_single_head=bool
                                    print_freq=int
                                    gpuid=[int]
        '''
        super(NormalNN, self).__init__()
        self.log = print if agent_config['print_freq'] > 0 else lambda \
            *args: None  # Use a void function to replace the print
        self.config = agent_config
        # If out_dim is a dict, there is a list of tasks. The model will have a head for each task.
        self.multihead = True if len(self.config['out_dim'])>1 else False  # A convenience flag to indicate multi-head/task
        if self.config['transfer_learning']:
            self.model = self.create_transfer_model()
        else:
            self.model = self.create_model()
        self.criterion_fn = nn.CrossEntropyLoss()
        if agent_config['gpuid'][0] >= 0:
            self.model.cuda()
            self.gpu = True
        else:
            self.gpu = False
        self.init_optimizer()
        self.reset_optimizer = False
        self.valid_out_dim = 'ALL'  # Default: 'ALL' means all output nodes are active
                                    # Set a interger here for the incremental class scenario
        self.transfer_loss_weight = 1.0
        self.accuracy_matrix = None
        self.is_1d = agent_config.get('is_1d', False)  # 添加一维数据标志
        self.input_dim = agent_config.get('input_dim', None)  # 添加输入维度参数

    def init_optimizer(self):
        optimizer_arg = {'params':self.model.parameters(),
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'

        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['schedule'],
                                                              gamma=0.1)

    def create_transfer_model(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](num_class=cfg['out_dim']['All'],
                                                                               base_net_type=cfg['transfer_backbone'],
                                                                               base_net=cfg['transfer_backbone_type'],
                                                                               transfer_loss=cfg['transfer_loss'])
        return model

    def create_1d_model(self):
        """创建一维数据的神经网络模型"""
        cfg = self.config
        layers = []
        input_dim = self.input_dim
        for hidden_dim in cfg.get('hidden_dims', [64, 32]):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, cfg['out_dim']['All']))
        return nn.Sequential(*layers)

    def create_model(self):
        cfg = self.config
        if self.is_1d:
            model = self.create_1d_model()
        else:
            model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']]()

        # Apply network surgery to the backbone
        # Create the heads for tasks (It can be single task or multi-task)
        n_feat = model.last.in_features

        # The output of the model will be a dict: {task_name1:output1, task_name2:output2 ...}
        # For a single-headed model the output will be {'All':output}
        model.last = nn.ModuleDict()
        for task, out_dim in cfg['out_dim'].items():
            model.last[task] = nn.Linear(n_feat, out_dim)

        # Redefine the task-dependent function
        def new_logits(self, x):
            outputs = {}
            for task, func in self.last.items():
                outputs[task] = func(x)
            return outputs

        # Replace the task-dependent function
        model.logits = MethodType(new_logits, model)
        # Load pre-trained weights
        if cfg['model_weights'] is not None:
            print('=> Load model weights:', cfg['model_weights'])
            model_state = torch.load(cfg['model_weights'],
                                     map_location=lambda storage, loc: storage)  # Load to CPU.
            model.load_state_dict(model_state)
            print('=> Load Done')
        return model

    def forward(self, x):
        return self.model.forward(x)

    def predict(self, inputs):
        self.model.eval()
        out = self.forward(inputs)
        for t in out.keys():
            out[t] = out[t].detach()
        return out

    def validation(self, dataloader):
        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = self.training
        self.eval()
        if self.config['transfer_learning']:
            for i, (input, target, task) in enumerate(dataloader):
                if self.gpu:
                    with torch.no_grad():
                        input = input.cuda()
                        target = target.cuda()
                output = self.model.predict(input)
                acc = accumulate_acc(output, target, task, acc)
        else:
            for i, (input, target, task) in enumerate(dataloader):
                if self.gpu:
                    with torch.no_grad():
                        input = input.cuda()
                        target = target.cuda()
                output = self.predict(input)
            # Summarize the performance of all tasks, or 1 task, depends on dataloader.
            # Calculated by total number of data.
                acc = accumulate_acc(output, target, task, acc)

        self.train(orig_mode)

        self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
              .format(acc=acc,time=batch_timer.toc()))
        return acc.avg

    def criterion(self, preds, targets, tasks, **kwargs):
        # The inputs and targets could come from single task or a mix of tasks
        # The network always makes the predictions with all its heads
        # The criterion will match the head and task to calculate the loss.
        if self.multihead:
            loss = 0
            for t, t_preds in preds.items():
                inds = [i for i in range(len(tasks)) if tasks[i]==t]  # The index of inputs that matched specific task
                if len(inds)>0:
                    t_preds = t_preds[inds]
                    t_target = targets[inds]
                    loss += self.criterion_fn(t_preds, t_target) * len(inds)  # restore the loss from average
            loss /= len(targets)  # Average the total loss by the mini-batch size
        else:
            pred = preds['All']
            if isinstance(self.valid_out_dim, int):  # (Not 'ALL') Mask out the outputs of unseen classes for incremental class scenario
                pred = preds['All'][:,:self.valid_out_dim]
            loss = self.criterion_fn(pred, targets)
        return loss

    def update_model(self, inputs, targets, tasks):
        out = self.forward(inputs)
        loss = self.criterion(out, targets, tasks)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach(), out

    def learn_batch(self, train_loader, val_loader=None):
        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_optimizer()

        for epoch in range(self.config['schedule'][-1]):
            data_timer = Timer()
            batch_timer = Timer()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            acc = AverageMeter()

            # Config the model and optimizer
            self.log('Epoch:{0}'.format(epoch))
            self.model.train()
            for param_group in self.optimizer.param_groups:
                self.log('LR:',param_group['lr'])

            # Learning with mini-batch
            data_timer.tic()
            batch_timer.tic()
            self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
            for i, (input, target, task) in enumerate(train_loader):

                data_time.update(data_timer.toc())  # measure data loading time

                if self.gpu:
                    input = input.cuda()
                    target = target.cuda()

                loss, output = self.update_model(input, target, task)
                input = input.detach()
                target = target.detach()
                self.scheduler.step()

                # measure accuracy and record loss
                acc = accumulate_acc(output, target, task, acc)
                losses.update(loss, input.size(0))

                batch_time.update(batch_timer.toc())  # measure elapsed time
                data_timer.toc()

                if ((self.config['print_freq']>0) and (i % self.config['print_freq'] == 0)) or (i+1)==len(train_loader):
                    self.log('[{0}/{1}]\t'
                          '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                          '{data_time.val:.4f} ({data_time.avg:.4f})\t'
                          '{loss.val:.3f} ({loss.avg:.3f})\t'
                          '{acc.val:.2f} ({acc.avg:.2f})'.format(
                        i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, acc=acc))

            self.log(' * Train Acc {acc.avg:.3f}'.format(acc=acc))

            # Evaluate the performance of current task
            if val_loader != None:
                self.validation(val_loader)

    def transfer_batch(self, source_train_loader, target_train_loader, source_name, val_loader=None):
        if self.reset_optimizer:
            self.log('Optimizer is reset!')
            self.init_optimizer()

        # load model weights
        if self.config['model_weights'] is not None:
            if int(source_name) in self.config['model_weight_list']:
                print('=> Load pretrained model weights:', self.config['model_weights']+source_name)
                model_state = torch.load(self.config['model_weights'] + source_name + '.pth',
                                         map_location=lambda storage, loc: storage)  # Load to CPU.
                self.model.load_state_dict(model_state, strict=False)
                print('=> Load Done')

        for epoch in range(self.config['schedule'][-1]):
            data_timer = Timer()
            batch_timer = Timer()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            clf_losses = AverageMeter()
            Transfer_losses = AverageMeter()
            acc = AverageMeter()

            source_iter = iter(source_train_loader)
            target_iter = iter(target_train_loader)

            # Config the model and optimizer
            self.log('Epoch:{0}'.format(epoch))
            self.model.train()
            for param_group in self.optimizer.param_groups:
                self.log('LR:', param_group['lr'])
                # Learning with mini-batch
                data_timer.tic()
                batch_timer.tic()
                self.log('Itr\t\t  Time\t\t\t    Data\t\t\t  Loss\t\t\t clf_loss\t   Acc')
                for i in range(len(target_train_loader)):

                    source_train_data = next(source_iter)
                    target_train_data = next(target_iter)

                    data_time.update(data_timer.toc())  # measure data loading time

                    if self.gpu:
                        source_train_label = source_train_data[1].cuda()
                        source_train_batch = source_train_data[0].cuda()
                        target_train_label = target_train_data[1].cuda()
                        target_train_batch = target_train_data[0].cuda()

                    clf_loss, loss, target_output = self.Transfer_update_model(source_train_batch,
                                                                               target_train_batch,
                                                                               source_train_label)
                    source_train_batch = source_train_batch.cuda()
                    self.scheduler.step()

                    # measure accuracy and record loss
                    acc = accumulate_acc(target_output, target_train_label, i, acc)
                    Transfer_losses.update(loss, source_train_batch.size(0))
                    clf_losses.update(clf_loss, source_train_batch.size(0))
                    target_output_label = torch.argmax(target_output, dim=1)

                    batch_time.update(batch_timer.toc())  # measure elapsed time
                    data_timer.toc()

                    if ((self.config['print_freq'] > 0) and (i % self.config['print_freq'] == 0)) or (i + 1) == len(
                            source_train_loader):
                        self.log('[{0}/{1}]\t'
                                 '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                                 '{data_time.val:.4f} ({data_time.avg:.4f})\t'
                                 '{loss.val:.3f} ({loss.avg:.3f})\t'
                                 '{clf_loss.avg:.3f}\t'
                                 '{acc.val:.2f} ({acc.avg:.2f})'.format(
                            i, len(source_train_loader), batch_time=batch_time,
                            data_time=data_time, loss=Transfer_losses, clf_loss=clf_losses, acc=acc))

                self.log(' * Train Acc {acc.avg:.3f}'.format(acc=acc))

                # Evaluate the performance of current task
                if val_loader != None:
                    self.validation(val_loader)

    def Transfer_update_model(self, source, target, source_label):
        clf_loss, transfer_loss, target_output = self.model.forward(source, target, source_label)
        loss = clf_loss + self.transfer_loss_weight * transfer_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return clf_loss, loss, target_output

    def pretrain(self, train_loader, val_loader, args):

        for epoch in range(args.schedule[-1]):
            pretrain_losses = AverageMeter()
            pretrain_acc = AverageMeter()

            self.log('Pretrain Epoch:{0}'.format(epoch))
            self.model.train()

            self.log('Itr\t\t  Data\t\t  Loss\t\tAcc')
            for i, (input, label, task) in enumerate(train_loader):
                input = input.cuda()
                label = label.cuda()

                loss, output = self.model.pretrain(input, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pretrain_acc = accumulate_acc(output, label, task, pretrain_acc)
                pretrain_losses.update(loss, input.size(0))

                if ((self.config['print_freq']>0) and (i % self.config['print_freq'] == 0)) or (i+1)==len(train_loader):
                    self.log('[{0}/{1}]\t'
                          '{loss.val:.3f} ({loss.avg:.3f})\t'
                          '{acc.val:.2f} ({acc.avg:.2f})'.format(
                        i, len(train_loader), loss=pretrain_losses, acc=pretrain_acc))

            if pretrain_acc.avg >99:
                val_acc = AverageMeter()
                self.model.eval()
                with torch.no_grad():
                    for j, (val_input, val_label, task) in enumerate(val_loader):
                        val_input = val_input.cuda()
                        val_label = val_label.cuda()
                        val_out = self.model.predict(val_input)
                        val_acc = accumulate_acc(val_out, val_label, task, val_acc)
                    if loss < 0.0001 and val_acc.avg == 100:
                        self.save_model(args.model_weights+task[0])
                        break
        # if pretrain_acc.avg > 99 and val_acc.avg > 99:
        #     break

    def learn_stream(self, data, label):
        assert False,'No implementation yet'

    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        if self.valid_out_dim == 'ALL':
            self.valid_out_dim = 0  # Initialize it with zero
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())

    def pesudo_label(self, target_dataloader, args, task_names, train_or_val='train'):
        # 由当前训练模型得到训练集所有样本的伪标签和伪标签的置信度，并保存为文件，供下一次迭代训练使用
        self.model.eval()
        with torch.no_grad():
            for i, (tar_input, tar_label, task) in enumerate(target_dataloader):
                tar_input = tar_input.cuda()
                tar_output = self.model.predict(tar_input)
                label_batch = torch.argmax(tar_output, dim=1)
                label_softmax = self.model.softmax(tar_output) # 用softmax的结果来表征标签的置信度
                label_confidence = label_softmax[torch.arange(label_softmax.size(0)), label_batch]
                if i == 0:
                    pes_label = torch.stack([label_batch, label_confidence], dim=1)
                else:
                    pes_label = torch.cat([pes_label,
                                           torch.stack([label_batch, label_confidence], dim=1)],
                                          dim=0)
        tasks_name = '-'.join(task_names)
        if os.path.exists(os.path.join(args.dataroot, 'pes_label', tasks_name)) is False:
            os.makedirs(os.path.join(args.dataroot, 'pes_label', tasks_name))
        torch.save(pes_label, os.path.join(args.dataroot, 'pes_label', tasks_name, train_or_val+'_'+task[0]+'.pt'))
        print('Pseudo label save done')


    def save_model(self, filename):
        model_state = self.model.state_dict()
        if isinstance(self.model, torch.nn.DataParallel):
            # Get rid of 'module' before the name of states
            model_state = self.model.module.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        print('=> Saving model to:', filename)
        torch.save(model_state, filename + '.pth')
        print('=> Save Done')

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

    def calculate_train_loss(self, data_loader):
        self.model.eval()  # 将模型设置为评估模式
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets, task in data_loader:  # 根据需要调整解包
                if self.gpu:  # 检查是否使用GPU
                    inputs = inputs.cuda()  # 将输入数据移动到GPU
                    targets = targets.cuda()  # 将目标数据移动到GPU
                    # 如果task在损失计算中需要，且是张量，也将其转移到GPU
                    task = task.cuda() if isinstance(task, torch.Tensor) else task

                outputs = {'All': self.model.predict(inputs)}
                loss = self.criterion(outputs, targets, task)
                total_loss += loss.item()
        return total_loss / len(data_loader)

    def calculate_accuracy(self, data_loader):
        self.model.eval()  # 将模型设置为评估模式
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets, _ in data_loader:  # 忽略第三个值
                if self.gpu:  # 检查是否使用GPU
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                outputs = self.model.predict(inputs)  # 使用模型的predict方法
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        return correct / total

    def select_target_domain(self, source_loader, target_loader, task_names, previous_task_loaders):
        # 初始化 accuracy_matrix
        T = len(task_names)
        if self.accuracy_matrix is None:
            self.accuracy_matrix = np.zeros((T, T))

        best_target_domain = None
        best_score = float('inf')

        # 初始化分布差异累积变量
        total_mmd_diff = 0.0
        total_kl_diff = 0.0
        num_batches = 0

        # 遍历多个 batch
        for source_data, _, _ in source_loader:
            target_data = next(iter(target_loader))[0]
            if self.gpu:
                source_data = source_data.cuda()
                target_data = target_data.cuda()

            # 计算分布差异指标
            mmd_diff, kl_diff = self.calculate_distribution_difference(source_data, target_data)
            total_mmd_diff += mmd_diff
            total_kl_diff += kl_diff
            num_batches += 1

            # 只使用前几个 batch 进行计算
            if num_batches >= 10:  # 例如，使用 10 个 batch
                break

        # 计算平均分布差异
        avg_mmd_diff = total_mmd_diff / num_batches
        avg_kl_diff = total_kl_diff / num_batches

        # 记录当前任务的准确率
        current_task_index = len(previous_task_loaders)
        self.accuracy_matrix[current_task_index, current_task_index] = self.validation(target_loader)

        # 记录之前任务的准确率
        for i, previous_task_loader in enumerate(previous_task_loaders):
            self.accuracy_matrix[current_task_index, i] = self.validation(previous_task_loader)

        # 计算后向迁移指标
        backward_transfer_impact = self.calculate_backward_transfer_impact(self.accuracy_matrix)

        # 检查 source_label 是否为 None
        source_labels = [data[1] for data in source_loader]
        if source_labels[0] is not None and self.gpu:
            source_labels = [label.cuda() for label in source_labels]

        # 计算前向迁移损失
        _, forward_transfer_loss, _ = self.model.forward(source_data, target_data, source_labels[0])

        # 计算融合指标
        fusion_score = self.fusion_metric(avg_mmd_diff, avg_kl_diff, backward_transfer_impact, forward_transfer_loss)

        # 打印指标
        print(f"Fusion score: {fusion_score:.4f}")
        print(f"Average MMD difference: {avg_mmd_diff:.4f}")
        print(f"Average KL divergence: {avg_kl_diff:.4f}")
        print(f"Forward transfer loss: {forward_transfer_loss:.4f}")
        print(f"Backward transfer impact: {backward_transfer_impact:.4f}")

        # 绘制准确率矩阵的混淆矩阵图
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.accuracy_matrix, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.xlabel("Target Task")
        plt.ylabel("Source Task")
        plt.title("Accuracy Matrix")

        # 生成图像文件名
        source_task_name = task_names[len(previous_task_loaders)]
        target_task_name = task_names[len(previous_task_loaders) + 1]
        file_name = f"accuracy_matrix_{source_task_name}_to_{target_task_name}.png"

        # 保存图像到文件
        plt.savefig(file_name)
        plt.close()

        # 选择具有最低融合指标的目标域
        if fusion_score < best_score:
            best_score = fusion_score
            best_target_domain = target_data

        return best_target_domain, fusion_score

    def calculate_distribution_difference(self, source_data, target_data):
        # 使用MMD和KL散度计算分布差异
        mmd_diff = self.calculate_mmd(source_data, target_data)
        kl_diff = self.calculate_kl_divergence(source_data, target_data)
        
        # 确保在进行任何 NumPy 操作之前将张量移动到 CPU
        if isinstance(mmd_diff, torch.Tensor):
            mmd_diff = mmd_diff.cpu().numpy()
        if isinstance(kl_diff, torch.Tensor):
            kl_diff = kl_diff.cpu().numpy()
        
        return mmd_diff, kl_diff

    def calculate_mmd(self, source_data, target_data, kernel='rbf', bandwidth=1.0):
        """
        计算源域和目标域之间的MMD。
        """
        def gaussian_kernel(x, y, bandwidth):
            x_size = x.size(0)
            y_size = y.size(0)
            dim = x.size(1)
            x = x.unsqueeze(1).expand(x_size, y_size, *x.shape[1:])
            y = y.unsqueeze(0).expand(x_size, y_size, *y.shape[1:])
            return torch.exp(-((x - y) ** 2).sum(dim=2) / (2 * bandwidth ** 2))

        if kernel == 'rbf':
            xx = gaussian_kernel(source_data, source_data, bandwidth)
            yy = gaussian_kernel(target_data, target_data, bandwidth)
            xy = gaussian_kernel(source_data, target_data, bandwidth)
            mmd = xx.mean() + yy.mean() - 2 * xy.mean()
            return mmd
        else:
            raise ValueError("Unsupported kernel type: {}".format(kernel))

    def calculate_kl_divergence(self, source_data, target_data):
        """
        计算源域和目标域之间的KL散度。
        """
        source_distribution = F.softmax(source_data, dim=1).mean(0)
        target_distribution = F.softmax(target_data, dim=1).mean(0)
        kl_div = entropy(source_distribution.cpu().numpy(), target_distribution.cpu().numpy())
        
        # 确保返回值是一个标量
        if isinstance(kl_div, np.ndarray):
            kl_div = np.sum(kl_div)
        
        return kl_div

    def calculate_backward_transfer_impact(self, accuracy_matrix):
        """
        计算后向迁移指标 (BWT)。
        :param accuracy_matrix: 一个 T x T 的矩阵，其中 R[i, j] 是模型在任务 j 上的测试准确率，在完成任务 i 后。
        :return: 后向迁移指标 (BWT) 的标量值。
        """
        T = accuracy_matrix.shape[0]
        bwt = 0.0
        for i in range(1, T):
            bwt += accuracy_matrix[T-1, i] - accuracy_matrix[i, i]
        bwt /= (T - 1)
        return bwt

    def fusion_metric(self, mmd_diff, kl_diff, backward_transfer_impact, forward_transfer_loss):
        # Ensure all inputs are scalars
        if isinstance(mmd_diff, np.ndarray):
            if mmd_diff.size == 1:
                mmd_diff = mmd_diff.item()
            else:
                mmd_diff = np.mean(mmd_diff)  # or another appropriate reduction
        if isinstance(kl_diff, np.ndarray):
            if kl_diff.size == 1:
                kl_diff = kl_diff.item()
            else:
                kl_diff = np.mean(kl_diff)  # or another appropriate reduction
        if isinstance(backward_transfer_impact, np.ndarray):
            if backward_transfer_impact.size == 1:
                backward_transfer_impact = backward_transfer_impact.item()
            else:
                backward_transfer_impact = np.mean(backward_transfer_impact)  # or another appropriate reduction
        if isinstance(forward_transfer_loss, np.ndarray):
            if forward_transfer_loss.size == 1:
                forward_transfer_loss = forward_transfer_loss.item()
            else:
                forward_transfer_loss = np.mean(forward_transfer_loss)  # or another appropriate reduction

        # 定义融合指标的计算方式
        w1, w2, w3, w4 = 100, 0.1, -5, -1  # 可以根据需要调整权重
        return w1 * mmd_diff + w2 * kl_diff + w3 * backward_transfer_impact + w4 * forward_transfer_loss

    def transfer_batch_1d(self, source_train_loader, target_train_loader, source_name, val_loader=None):
        if self.reset_optimizer:
            self.log('Optimizer is reset!')
            self.init_optimizer()

        if self.config['model_weights'] is not None:
            if int(source_name) in self.config['model_weight_list']:
                print('=> Load pretrained model weights:', self.config['model_weights']+source_name)
                model_state = torch.load(self.config['model_weights'] + source_name + '.pth',
                                         map_location=lambda storage, loc: storage)
                self.model.load_state_dict(model_state, strict=False)
                print('=> Load Done')

        for epoch in range(self.config['schedule'][-1]):
            data_timer = Timer()
            batch_timer = Timer()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            clf_losses = AverageMeter()
            Transfer_losses = AverageMeter()
            acc = AverageMeter()

            source_iter = iter(source_train_loader)
            target_iter = iter(target_train_loader)

            self.log('Epoch:{0}'.format(epoch))
            self.model.train()
            for param_group in self.optimizer.param_groups:
                self.log('LR:', param_group['lr'])
                data_timer.tic()
                batch_timer.tic()
                self.log('Itr\t\t  Time\t\t\t    Data\t\t\t  Loss\t\t\t clf_loss\t   Acc')
                for i in range(len(target_train_loader)):

                    source_train_data = next(source_iter)
                    target_train_data = next(target_iter)

                    data_time.update(data_timer.toc())

                    if self.gpu:
                        source_train_label = source_train_data[1].cuda()
                        source_train_batch = source_train_data[0].cuda()
                        target_train_label = target_train_data[1].cuda()
                        target_train_batch = target_train_data[0].cuda()

                    clf_loss, loss, target_output = self.Transfer_update_model(source_train_batch,
                                                                               target_train_batch,
                                                                               source_train_label)
                    source_train_batch = source_train_batch.cuda()
                    self.scheduler.step()

                    acc = accumulate_acc(target_output, target_train_label, i, acc)
                    Transfer_losses.update(loss, source_train_batch.size(0))
                    clf_losses.update(clf_loss, source_train_batch.size(0))
                    target_output_label = torch.argmax(target_output, dim=1)

                    batch_time.update(batch_timer.toc())
                    data_timer.toc()

                    if ((self.config['print_freq'] > 0) and (i % self.config['print_freq'] == 0)) or (i + 1) == len(
                            source_train_loader):
                        self.log('[{0}/{1}]\t'
                                 '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                                 '{data_time.val:.4f} ({data_time.avg:.4f})\t'
                                 '{loss.val:.3f} ({loss.avg:.3f})\t'
                                 '{clf_loss.avg:.3f}\t'
                                 '{acc.val:.2f} ({acc.avg:.2f})'.format(
                            i, len(source_train_loader), batch_time=batch_time,
                            data_time=data_time, loss=Transfer_losses, clf_loss=clf_losses, acc=acc))

                self.log(' * Train Acc {acc.avg:.3f}'.format(acc=acc))

                if val_loader != None:
                    self.validation_1d(val_loader)

    def validation_1d(self, dataloader):
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = self.training
        self.eval()
        if self.config['transfer_learning']:
            for i, (input, target, task) in enumerate(dataloader):
                if self.gpu:
                    with torch.no_grad():
                        input = input.cuda()
                        target = target.cuda()
                output = self.model.predict(input)
                acc = accumulate_acc(output, target, task, acc)
        else:
            for i, (input, target, task) in enumerate(dataloader):
                if self.gpu:
                    with torch.no_grad():
                        input = input.cuda()
                        target = target.cuda()
                output = self.predict(input)
                acc = accumulate_acc(output, target, task, acc)

        self.train(orig_mode)

        self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
              .format(acc=acc,time=batch_timer.toc()))
        return acc.avg

def accumulate_acc(output, target, task, meter):
    if output is dict:
        if 'All' in output.keys(): # Single-headed model
            meter.update(accuracy(output['All'], target), len(target))
        else:  # outputs from multi-headed (multi-task) model
            for t, t_out in output.items():
                inds = [i for i in range(len(task)) if task[i] == t]  # The index of inputs that matched specific task
                if len(inds) > 0:
                    t_out = t_out[inds]
                    t_target = target[inds]
                    meter.update(accuracy(t_out, t_target), len(inds))
    else:
        meter.update(accuracy(output, target), len(target))

    return meter