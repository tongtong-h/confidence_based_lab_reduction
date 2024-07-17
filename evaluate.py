# +
import torch

from utils import get_metrics


# -

def evaluate_select(cur_results_dict, ground_truth_dict):    
    select_out = cur_results_dict['Selections']
    normal_out = cur_results_dict['Normal']
    stable_out = cur_results_dict['Stable']
    
        
    reduce_mask = cur_results_dict['Reduce mask']
    auto_mask = cur_results_dict['Auto mask']
    
    normal_mask = ground_truth_dict['Normal mask'][:,1:]
    stable_mask = ground_truth_dict['Stable mask'][:,1:]
    targets = ground_truth_dict['Targets'][:,1:]
    
    valid_mask = (stable_mask>=0)
    reduce_rate = torch.sum(reduce_mask[valid_mask>0]) / torch.sum(reduce_mask[valid_mask>0]!=np.nan)
    
    coverage_rate = select_out[valid_mask_group>0].mean()
    
    mask = (valid_mask==1)&(reduce_mask==1)
    
    prob_thres = 0.5
    normal_metrics = get_metrics(normal_out[mask],  normal_mask[mask], 
                                 thres=prob_thres)
    stable_metrics = get_metrics(stable_out[mask],  stable_mask[mask], 
                                 thres=prob_thres)

    
    return {
        'normal_acc': normal_metrics['Acc'],
        'stable_acc': stable_metrics['Acc'],
        'coverage': coverage_rate,
        'reduction': reduce_rate,
    }


def evaluate_select_all(model_path, reduction_mode='no_reduction'):
    model_acc_all = []

    for kf in range(5):
        print(f'kfold: {kf}')
        test_ground_truth = pickle.load(open(f'output/test_ground_truth_kf{kf}.pkl', 'rb'))

        model_acc_kf = []

        for coverage in np.arange(0.75,1.1,0.05):
            coverage = round(coverage, 2)

            try:
                test_pred_results = pickle.load(open(model_path, 'rb'))
                cov_acc = evaluate_select(test_pred_results, test_ground_truth)
                model_acc_kf.append(cov_acc)
            except Exception as e:
                print(e)
                continue

        model_acc_all.append(model_acc_kf)


    fig, axs = plt.subplots(1,2, figsize=(8,4))
    axs = axs.reshape(-1)
    plt.subplots_adjust(wspace=0.4, hspace=0.5)

    metric_label = ['Hgb Normality\nAccuracy','Hgb Stability\nAccuracy']

    for mi, metric in enumerate(['normal_acc','stable_acc']):
        model_score_list_all = []
        model_red_list_all = []
        for model_acc in model_acc_all:
            model_score_list = [acc_dict[metric] for acc_dict in model_acc]
            model_red_list = [acc_dict['reduction'] for acc_dict in model_acc]
            model_score_list_all.append(model_score_list)
            model_red_list_all.append(model_red_list)


        model_score_list_all = np.array(model_score_list_all).mean(axis=0)
        model_red_list_all = np.array(model_red_list_all).mean(axis=0)


        axs[mi].plot(
            model_red_list_all,
            model_score_list_all,
            '*-'
        )



        axs[mi].set_ylim(top=1.0, bottom=0.79)
        axs[mi].set_xlabel('Proportion of selected panels')
        axs[mi].set_ylabel(metric_label[mi])
        axs[mi].grid()

    plt.show()
