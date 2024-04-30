import torch


def average_models(model_paths):
    averaged_params = [{}]

    for i, model_path in enumerate(model_paths):
        model = torch.load(model_path)

        if len(averaged_params) <= i + 1:
            averaged_params.append({})

        for param_name in model.keys():
            if param_name.startswith('module.'):
                if param_name.startswith('module.GFF.'):
                    param_value = model[param_name]
                    if param_name not in averaged_params[i]:
                        averaged_params[i][param_name] = param_value.clone()
                    else:
                        averaged_params[i][param_name] += param_value
                else:
                    averaged_params[i][param_name] = param_value.clone()

    for i in range(len(model_paths)):
        for param_name in averaged_params[i].keys():
            if param_name.startswith('module.GFF.'):
                averaged_params[i][param_name] /= len(model_paths)
        averaged_model = model.copy()
        for param_name in averaged_params[i].keys():
            averaged_model[param_name] = averaged_params[i][param_name]
        torch.save(averaged_model, f"AvgModel/averaged_model{i+1}.pth")


# 定义五个模型的文件路径
model_paths = ['experiments/Cam1_experiments/epochs/Cam1_best_ckp.pth',
               'experiments/Cam2_experiments/epochs/Cam2_best_ckp.pth',
               'experiments/Cam3_experiments/epochs/Cam3_best_ckp.pth',
               'experiments/Cam4_experiments/epochs/Cam4_best_ckp.pth',
               'experiments/Cam5_experiments/epochs/Cam5_best_ckp.pth']

# 执行模型平均并保存结果
average_models(model_paths)

print("==========Avg Finish==========")

# import torch
#
#
# # 定义模型平均的函数
# def average_models(model_paths):
#     # 初始化参数字典
#     averaged_params = {}
#
#     # 加载每个模型的参数并累加
#     for model_path in model_paths:
#         model = torch.load(model_path)
#
#         for param_name in model.keys():
#             if param_name.startswith('module.'):
#                 param_value = model[param_name]
#                 if param_name not in averaged_params:
#                     averaged_params[param_name] = param_value.clone()
#                 else:
#                     averaged_params[param_name] += param_value
#
#     # 计算平均值
#     for param_name in averaged_params.keys():
#         averaged_params[param_name] /= len(model_paths)
#
#     # 保存平均结果
#     averaged_model = model.copy()
#     for param_name in averaged_params.keys():
#         averaged_model[param_name] = averaged_params[param_name]
#     torch.save(averaged_model, 'AvgModel/averaged_model1.pth')
#
#
# # 定义五个模型的文件路径
# model_paths = ['experiments/Cam1_experiments/epochs/Cam1_best_ckp.pth',
#                'experiments/Cam2_experiments/epochs/Cam2_best_ckp.pth',
#                'experiments/Cam3_experiments/epochs/Cam3_best_ckp.pth',
#                'experiments/Cam4_experiments/epochs/Cam4_best_ckp.pth',
#                'experiments/Cam5_experiments/epochs/Cam5_best_ckp.pth']
#
# # 执行模型平均并保存结果
# average_models(model_paths)
#
# print("==========Avg Finish==========")