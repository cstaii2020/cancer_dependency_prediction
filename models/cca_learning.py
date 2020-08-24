import numpy as np
import torch
from scipy.stats import pearsonr
import os
import pandas as pd

from models.base_models import LearningModel
from models.cca import scale_train_test, evaluate_cca_wa_wb, unscale_prediction

from sklearn.cross_decomposition import CCA


class CcaClassifier(LearningModel):
    def combined_eval(self, evaluations):
        all_scores = np.array(evaluations)
        average_scores = all_scores.mean(axis=0)
        score_data = np.vstack((all_scores, average_scores))
        score_df = pd.DataFrame(columns=self.class_names,
                                index=list(range(len(all_scores)))+["avg"], data=score_data)

        count = 1
        file_name = self.out_name
        while os.path.exists(file_name):
            file_name = f"{file_name}_({count})"
            count += 1

        score_df.to_csv(file_name)
        return score_df

    def train_eval(self, train_index, test_index):
        train_source, test_source = self.source[train_index], self.source[test_index]
        train_target, test_target = self.target[train_index], self.target[test_index]

        train_source, test_source = scale_train_test(train_source, test_source)
        train_target, _ = scale_train_test(train_target, test_target)

        # rho, w_t, w_s, _ = evaluate_cca_wa_wb(train_target, train_source)
        cca = CCA(n_components=min(train_source.shape[1], train_target.shape[1]), max_iter=1000)
        cca.fit(train_source, train_target)
        w_s = cca.x_rotations_
        w_t = cca.y_rotations_

        predicted_target = test_source @ w_s @ np.linalg.pinv(w_t)
        predicted_target = unscale_prediction(train_target, predicted_target)

        if self.target_encoder is not None:
            test_target = self.original_target[test_index]
            predicted_target = self.target_encoder.decode(torch.as_tensor(predicted_target)).detach().numpy()

        scores = np.zeros(self.original_target.shape[1])
        for i in range(self.original_target.shape[1]):
            predicted = predicted_target[:, i]
            actual = test_target[:, i]
            r, pval = pearsonr(predicted, actual)
            scores[i] = r

        return scores
