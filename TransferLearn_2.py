import os
import dataloaders.base
from dataloaders.datasetGen import PermutedGen, SplitGen
import argparse
import agents
from random import shuffle
from collections import OrderedDict
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt

# Define global variables for metrics
train_losses = []
source_accuracies = []
target_accuracies = []

def get_args(argv):
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")

    # model backbone
    parser.add_argument('--model_type', type=str, default='Transfernet',
                        help="The type (mlp|lenet|vgg|resnet) of backbone network. If transfer learning, use Transfernet")
    parser.add_argument('--model_name', type=str, default='TransferNet', help="The name of actual model for the backbone,"
                                                                              "If transfer learning, use TransferNet")
    parser.add_argument('--transfer_backbone', type=str, default='resnet',
                        help="The name of backbone for transfer learning(model .py file name)")
    parser.add_argument('--transfer_backbone_type', type=str, default='ResNet_Transfer10',
                        help="The name of actual backbone for transfer learning(model class name)")
    parser.add_argument('--force_out_dim', type=int, default=4,
                        help="Set 0 to let the task decide the required output dimension")

    # icemental&transfer model
    parser.add_argument('--agent_type', type=str, default='default', help="The type (filename) of agent")
    parser.add_argument('--agent_name', type=str, default='NormalNN', help="The class name of agent")
    parser.add_argument('--transfer_learning', default=True, action='store_true')
    parser.add_argument('--transfer_loss', type=str, default='wd', help='loss type of transfer learning')


    # dataset
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--dataset', type=str, default='MNIST', help="MNIST(default)|CIFAR10|CIFAR100")
    parser.add_argument('--n_permutation', type=int, default=10, help="Enable permuted tests when >0")
    parser.add_argument('--first_split_size', type=int, default=4)
    parser.add_argument('--other_split_size', type=int, default=4)
    parser.add_argument('--no_class_remap', dest='no_class_remap', default=False, action='store_true',
                        help="Avoid the dataset with a subset of classes doing the remapping. Ex: [2,5,6 ...] -> [0,1,2 ...]")
    parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                        help="Allow data augmentation during training")
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                        help="Randomize the classes in splits")
    parser.add_argument('--rand_split_order', dest='rand_split_order', default=False, action='store_true',
                        help="Randomize the order of splits")
    parser.add_argument('--incremental_class', dest='incremental_class', default=False, action='store_true',
                        help="The number of output node in the single-headed model increases along with new categories.")
    parser.add_argument('--pseudo_label', default=True)

    # training process
    parser.add_argument('--task_order', type=int, default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--workers', type=int, default=5, help="#Thread for dataloader")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--schedule', nargs="+", type=int, default=[1],
                        help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number is the end epoch")
    parser.add_argument('--print_freq', type=float, default=100, help="Print the log at every x iteration")
    parser.add_argument('--model_weights', type=str, default=None,
                        help="The path to the file for the model weights (*.pth).")
    parser.add_argument('--model_weight_list', type=list, default=[1],
                        help="The name list of the pretrained model weights")
    parser.add_argument('--reg_coef', nargs="+", type=float, default=[0.],
                        help="The coefficient for regularization. Larger means less plasilicity. Give a list for hyperparameter search.")
    parser.add_argument('--eval_on_train_set', dest='eval_on_train_set', default=False, action='store_true',
                        help="Force the evaluation on train set")
    parser.add_argument('--offline_training', dest='offline_training', default=False, action='store_true',
                        help="Non-incremental learning by make all data available in one batch. For measuring the upperbound performance.")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--threshold', type=float, default=50, help="Threshold for determining if a target domain should be skipped due to high fusion score.")
    parser.add_argument('--is_1d', type=bool, default=False, help="Whether to use 1D data")
    parser.add_argument('--input_dim', type=int, default=None, help="Input dimension for 1D data")
    parser.add_argument('--save_checkpoints', dest='save_checkpoints', default=False, action='store_true',
                        help="Whether to save checkpoints")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Directory to save checkpoints")

    args = parser.parse_args(argv)
    return args


def TransferRun(args):
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    # prepare dataloader for transfer learning
    train_dataset, val_dataset = dataloaders.base.__dict__[args.dataset](args.dataroot, args.train_aug)

    train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset, val_dataset,
                                                                           first_split_sz=args.first_split_size,
                                                                           other_split_sz=args.other_split_size,
                                                                           rand_split=args.rand_split,
                                                                           remap_class=not args.no_class_remap)

    # Prepare the Agent (model)
    agent_config = {'lr': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay,
                    'schedule': args.schedule,
                    'model_type': args.model_type, 'model_name': args.model_name, 'model_weights': args.model_weights,
                    'out_dim': {'All': args.force_out_dim} if args.force_out_dim > 0 else task_output_space,
                    'optimizer': args.optimizer,
                    'print_freq': args.print_freq, 'gpuid': args.gpuid,
                    'reg_coef': args.reg_coef,
                    'transfer_learning': args.transfer_learning,
                    'transfer_backbone': args.transfer_backbone,
                    'transfer_backbone_type': args.transfer_backbone_type,
                    'transfer_loss': args.transfer_loss,
                    'model_weight_list': args.model_weight_list,
                    'is_1d': args.is_1d,
                    'input_dim': args.input_dim}
    agent = agents.__dict__[args.agent_type].__dict__[args.agent_name](agent_config)
    print(agent.model)
    print('#Transfer Loss Type:', args.transfer_loss)
    print('#parameter of model:', agent.count_parameter())

    # Decide split ordering
    task_names = sorted(list(task_output_space.keys()), key=int)
    if args.rand_split_order:
        shuffle(task_names)
        print('Shuffled task order:', task_names)
    elif len(args.task_order) != 0:
        task_names = [task_names[i - 1] for i in args.task_order]
    print('Task order:', task_names)

    acc_table = OrderedDict()

    # Initialize lists to store metrics
    train_losses.clear()
    source_accuracies.clear()
    target_accuracies.clear()

    skipped_tasks = []  # List to store skipped tasks

    # 初始化之前任务的数据加载器列表
    previous_task_loaders = []

    if args.offline_training:  # Non-incremental learning / offline_training / measure the upper-bound performance
        for i in range(len(task_names)-1):
            source_train_name = task_names[i]
            target_train_name = task_names[i+1]
            print('offline==================', source_train_name, '→', target_train_name, '===================offline')

            source_train_loader = torch.utils.data.DataLoader(train_dataset_splits[source_train_name],
                                                              batch_size=args.batch_size,
                                                              shuffle=True, num_workers=args.workers)
            source_val_loader = torch.utils.data.DataLoader(val_dataset_splits[source_train_name],
                                                            batch_size=args.batch_size,
                                                            shuffle=False, num_workers=args.workers)
            target_train_loader = torch.utils.data.DataLoader(train_dataset_splits[target_train_name],
                                                              batch_size=args.batch_size, shuffle=True,
                                                              num_workers=args.workers)
            target_val_loader = torch.utils.data.DataLoader(val_dataset_splits[target_train_name],
                                                            batch_size=args.batch_size, shuffle=False,
                                                            num_workers=args.workers)
            if args.transfer_learning:
                if os.path.exists(args.model_weights+task_names[i]+'.pth') is False:
                    agent.pretrain(source_train_loader, source_val_loader, args)

            # 添加当前任务的训练数据加载器到列表中
            previous_task_loaders.append(source_train_loader)

            # 先进行迁移训练
            if args.is_1d:
                agent.transfer_batch_1d(source_train_loader, target_train_loader, source_train_name, target_val_loader)
            else:
                agent.transfer_batch(source_train_loader, target_train_loader, source_train_name, target_val_loader)

            # Evaluate
            acc_table[source_train_name] = OrderedDict()
            for j in range(i + 1):
                val_name = task_names[j]
                print('validation split name:', val_name)
                val_data = val_dataset_splits[val_name] if not args.eval_on_train_set else train_dataset_splits[
                    val_name]
                val_loader = torch.utils.data.DataLoader(val_data,
                                                         batch_size=args.batch_size, shuffle=False,
                                                         num_workers=args.workers)
                acc_table[val_name][source_train_name] = agent.validation(val_loader)

            # Calculate metrics
            train_loss = agent.calculate_train_loss(source_train_loader)
            source_accuracy = agent.calculate_accuracy(source_val_loader)
            target_accuracy = agent.calculate_accuracy(target_val_loader)

            # Append the calculated metrics
            train_losses.append(train_loss)
            source_accuracies.append(source_accuracy)
            target_accuracies.append(target_accuracy)

            # Select target domain and evaluate fusion score
            selected_target, fusion_score = agent.select_target_domain(source_train_loader, target_train_loader, task_names, previous_task_loaders)

            # Check fusion score and decide whether to save results
            if selected_target is None or fusion_score > args.threshold:
                print(f"Skipping target domain {target_train_name} due to high fusion score.")
                skipped_tasks.append(target_train_name)
                # 不保存伪标签和checkpoints，直接继续下一次循环
                continue

            # 只有在不跳过的情况下才保存伪标签
            agent.pesudo_label(target_train_loader, args, task_names, train_or_val='train')
            agent.pesudo_label(target_val_loader, args, task_names, train_or_val='val')

            # 可以在这里添加保存checkpoints的代码
            if args.save_checkpoints:
                checkpoint_name = f"checkpoint_{source_train_name}_to_{target_train_name}.pth"
                agent.save_model(os.path.join(args.checkpoint_dir, checkpoint_name))

    else:  # Incremental transfer learning
        i = 0  # 源域索引
        while i < len(task_names) - 1:
            source_train_name = task_names[i]
            target_idx = i + 1  # 目标域索引
            
            while target_idx < len(task_names):
                target_train_name = task_names[target_idx]
                tasks_name = '-'.join(task_names)

                print('======================', source_train_name, '→', target_train_name, '=======================')
                
                # label replace 伪标签替换，使用上一次训练的目标域伪标签，替换本次训练的源域标签
                if args.pseudo_label and i != 0:
                    if os.path.exists(os.path.join(args.dataroot, 'pes_label', tasks_name)):
                        train_pseudo_label = torch.load(os.path.join(args.dataroot, 'pes_label', tasks_name,
                                                                     'train_'+task_names[i]+'.pt'),
                                                        map_location=torch.device('cpu'))
                        train_wrapped_dataset = train_dataset_splits[source_train_name]
                        train_wrapped_dataset.dataset.dataset.label_replace(train_pseudo_label)
                        print('Pseudo label load done')
                    else:
                        print('Task' + task_names[i] +'does not have pseudo label')
                        train_wrapped_dataset = train_dataset_splits[source_train_name]
                else:
                    train_wrapped_dataset = train_dataset_splits[source_train_name]

                source_train_loader = torch.utils.data.DataLoader(train_wrapped_dataset,
                                                                  batch_size=args.batch_size,
                                                                  shuffle=True, num_workers=args.workers)
                source_val_loader = torch.utils.data.DataLoader(val_dataset_splits[source_train_name],
                                                                batch_size=args.batch_size,
                                                                shuffle=False, num_workers=args.workers)
                target_train_loader = torch.utils.data.DataLoader(train_dataset_splits[target_train_name],
                                                                  batch_size=args.batch_size, shuffle=True,
                                                                  num_workers=args.workers)
                target_val_loader = torch.utils.data.DataLoader(val_dataset_splits[target_train_name],
                                                                batch_size=args.batch_size, shuffle=False,
                                                                num_workers=args.workers)

                if args.transfer_learning:
                    if os.path.exists(args.model_weights+task_names[i]+'.pth') is False:
                        agent.pretrain(source_train_loader, source_val_loader, args)

                # 添加当前任务的训练数据加载器到列表中
                if source_train_loader not in previous_task_loaders:
                    previous_task_loaders.append(source_train_loader)

                # 先进行迁移训练
                if args.is_1d:
                    agent.transfer_batch_1d(source_train_loader, target_train_loader, source_train_name, target_val_loader)
                else:
                    agent.transfer_batch(source_train_loader, target_train_loader, source_train_name, target_val_loader)

                # Evaluate
                acc_table[source_train_name] = OrderedDict()
                for j in range(i+1):
                    val_name = task_names[j]
                    print('validation split name:', val_name)
                    val_data = val_dataset_splits[val_name] if not args.eval_on_train_set else train_dataset_splits[val_name]
                    val_loader = torch.utils.data.DataLoader(val_data,
                                                             batch_size=args.batch_size, shuffle=False,
                                                             num_workers=args.workers)
                    acc_table[val_name][source_train_name] = agent.validation(val_loader)

                # Calculate metrics
                train_loss = agent.calculate_train_loss(source_train_loader)
                source_accuracy = agent.calculate_accuracy(source_val_loader)
                target_accuracy = agent.calculate_accuracy(target_val_loader)

                # Append the calculated metrics
                train_losses.append(train_loss)
                source_accuracies.append(source_accuracy)
                target_accuracies.append(target_accuracy)

                # Select target domain and evaluate fusion score
                selected_target, fusion_score = agent.select_target_domain(source_train_loader, target_train_loader, task_names, previous_task_loaders)

                # Check fusion score and decide whether to try next target domain
                if selected_target is None or fusion_score > args.threshold:
                    print(f"Skipping target domain {target_train_name} due to high fusion score.")
                    skipped_tasks.append(target_train_name)
                    target_idx += 1  # 尝试下一个目标域
                    continue  # 跳过当前目标域，尝试下一个

                # 只有在不跳过的情况下才保存伪标签
                agent.pesudo_label(target_train_loader, args, task_names, train_or_val='train')
                agent.pesudo_label(target_val_loader, args, task_names, train_or_val='val')

                # 可以在这里添加保存checkpoints的代码
                if args.save_checkpoints:
                    checkpoint_name = f"checkpoint_{source_train_name}_to_{target_train_name}.pth"
                    agent.save_model(os.path.join(args.checkpoint_dir, checkpoint_name))
                
                # 成功处理了一个目标域，更新源域索引
                i = target_idx
                break  # 退出内层循环，更新源域
            
            # 如果已经尝试了所有可能的目标域都不成功，也需要移动到下一个源域
            if target_idx >= len(task_names):
                i += 1

    # Handle skipped tasks
    for skipped_task in skipped_tasks:
        print(f"Revisiting skipped task: {skipped_task}")
        # Implement logic to revisit and train on skipped tasks

    return acc_table, task_names

# Plotting the metrics after training
def plot_metrics():
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # Plot training loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    # Plot source domain accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, source_accuracies, label='Source Domain Accuracy', color='g')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Source Domain Accuracy')
    plt.legend()
    
    # Plot target domain accuracy
    plt.subplot(1, 3, 3)
    plt.plot(epochs, target_accuracies, label='Target Domain Accuracy', color='r')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Target Domain Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    setattr(args, 'dataset', 'ZXJ_GD')
    setattr(args, 'dataroot', r'D:\DATABASE\ZXJ_GD\var_speed_sample\Continual_Learning\cmor_test')
    setattr(args, 'no_class_remap', False)
    setattr(args, 'n_permutation', 0)

    setattr(args, 'transfer_loss', 'bnm')
    setattr(args, 'agent_name', 'GEM_1100')
    setattr(args, 'agent_type', 'customization')
    setattr(args, 'model_weights', './models/pretrain/cmor/resnet/')

    setattr(args, 'offline_training', False)
    setattr(args, 'task_order', [1, 2, 3, 4, 5, 6, 7])

    reg_coef_list = args.reg_coef
    avg_final_acc = {}

    # The for loops over hyper-paramerters or repeats
    for reg_coef in reg_coef_list:
        args.reg_coef = reg_coef
        avg_final_acc[reg_coef] = np.zeros(args.repeat)
        for r in range(args.repeat):
            print(args.agent_type)
            print(args.agent_name)

            # Run the experiment
            acc_table, task_names = TransferRun(args)
            print(acc_table)

            # Calculate average performance across tasks
            # Customize this part for a different performance metric
            avg_acc_history = [0] * len(task_names)
            for i in range(len(task_names)-1):
                train_name = task_names[i]
                cls_acc_sum = 0
                for j in range(i + 1):
                    val_name = task_names[j]
                    cls_acc_sum += acc_table[val_name][train_name]
                avg_acc_history[i] = cls_acc_sum / (i + 1)
                print('Task', train_name, 'average acc:', avg_acc_history[i])

            # Gather the final avg accuracy
            avg_final_acc[reg_coef][r] = avg_acc_history[-1]

            # Print the summary so far
            print('===Summary of experiment repeats:', r + 1, '/', args.repeat, '===')
            print('The regularization coefficient:', args.reg_coef)
            print('The last avg acc of all repeats:', avg_final_acc[reg_coef])
            print('mean:', avg_final_acc[reg_coef].mean(), 'std:', avg_final_acc[reg_coef].std())
    for reg_coef, v in avg_final_acc.items():
        print('reg_coef:', reg_coef, 'mean:', avg_final_acc[reg_coef].mean(), 'std:', avg_final_acc[reg_coef].std())

    # Call the function to plot metrics
    plot_metrics()
