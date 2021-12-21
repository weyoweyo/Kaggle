

def save_data(submission_file, Y_pred, path, file_name, ext=".csv"):
    submission_file.iloc[:, 1] = Y_pred
    submission_file.to_csv(path + file_name + ext, index=False)
