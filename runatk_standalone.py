from models import *
import shap
from sklearn.metrics import accuracy_score, confusion_matrix

#preprocess data
#load shap dicts
#load models
#run mp function
#save



def shap_attack_mp(*args): # this is a self contained function that can be run in parallel, shap values are computed in real time here for each sample
    topf, shap_importance, model, neg_data_test, pos_data_test, xtest, ytest, tree_fps, tree_fns, fps, fns, increase_fn, model_type = args

    FP_count = []
    FN_count = []
    Accuracy = []
    x_copy = xtest.copy()
    vis_dataset = xtest.copy()
    precision = 10
    increase_fn = increase_fn
    print('Loaded args')
    if model_type == 'deep':
        preds = df_dl_predict(model, x_copy, ytest, ds_type='tcga', device='cpu')
        explainer = shap.DeepExplainer(model)
    else:
        preds = model.predict(x_copy)
        explainer = shap.TreeExplainer(model)

    confm_upd = confusion_matrix(ytest, preds)
    FP_count.append(confm_upd[0][1])
    FN_count.append(confm_upd[1][0])
    Accuracy.append(accuracy_score(ytest, preds))
    
    
    top_feat = shap_importance.iloc[:topf]
    top_feat_names = top_feat.col_name.values
    zero_idx_test = neg_data_test.index
    one_idx_test = pos_data_test.index

    x_copy = xtest.copy()  # get a fresh set
    print('Running SHAP attack...')

    for index, row in x_copy.iterrows():  # FP
        
        if model_type == 'deep':
            shapvalues = explainer.shap_values(np.array(row).unsqueeze(1).to(device).to(torch.float32))
            shap_arr = shapvalues.reshape(shapvalues.shape[0],1,-1)
            shap_arr = shap_arr.transpose(0,2,1).squeeze()
            shap_importance = shap.Explanation(shap_arr,  
                    base_values,
                    data=vis_dataset.values,
                    feature_names=vis_dataset.columns)     
        else:
            shap_importance = explainer.shap_values(row)
            
        top_feat = shap_importance.iloc[:topf]
        top_feat_names = top_feat.col_name.values
        
        
        if index in zero_idx_test:

            id_neg = tree_fps.query(row, k=2)

            if id_neg[0][
                0] == 0:  # this means we are getting fps vector TODO: remove fps from the test set before running the algorithm
                continue
            else:
                vector_id = id_neg[1][0]
                a0 = row
                a1 = fps.iloc[vector_id]

                for feature in top_feat_names:
                    a0_val = a0[feature]
                    a1_val = a1[feature]
                    sample_space = np.linspace(a0_val, a1_val, precision)
                    x_copy.at[index, feature] = sample_space[-2]
                    # print(f"Feature: {feature} | Original vector value: {a0_val} | Closest FP vector value {a1_val} | Sample space: {sample_space[-2]}")

        if increase_fn:
            if index in one_idx_test:  # positive labels or 1

                id_neg = tree_fns.query(row, k=2)
                # print(id_neg)
                if id_neg[0][
                    0] == 0:  # this means we are getting fps vector TODO: remove fns from the test set before running the algorithm
                    continue
                else:
                    vector_id = id_neg[1][0]
                    # print(id_neg)

                    a0 = row
                    a1 = fns.iloc[vector_id]

                for feature in top_feat_names:
                    a0_val = a0[feature]
                    a1_val = a1[feature]
                    sample_space = np.linspace(a0_val, a1_val, precision)

                    # print(f"Feature: {feature} | Original vector value: {a0_val} | Closest FN vector value {a1_val} | Sample space: {sample_space[-2]}")

                    x_copy.at[index, feature] = sample_space[-2]

    if model_type == 'deep':
        preds = df_dl_predict(model, x_copy, ytest, ds_type='tcga', device='cpu')
    else:
        preds = model.predict(x_copy)
    # preds=df_dl_predict(model, x_copy,ytest,ds_type='tcga')
    confm_upd = confusion_matrix(ytest, preds)
    FP_count.append(confm_upd[0][1])
    FN_count.append(confm_upd[1][0])
    Accuracy.append(accuracy_score(ytest, preds))
    print(
        f'Analysis using top {topf} features completed! FP: {confm_upd[0][1]} FN: {confm_upd[1][0]} Acc: {accuracy_score(ytest, preds)}')
    print('---------------------------------------------')
    return FP_count, FN_count, Accuracy, x_copy


if __name__ == '__main__':
    pass