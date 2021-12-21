from datetime import datetime
from utils import dataset
from models.knn import knn_pipeline
import warnings
warnings.filterwarnings('ignore')

print("knn model comparison...")
start_time = datetime.now()
print(start_time)

train, raw_test, sub = dataset.get_from_files('train', 'test_nolabels', 'sample_submission')
features, labels = dataset.get_train(train)

knn_pipeline.run(features, labels, 'knn comparison original')

print("end")
end_time = datetime.now()
print(end_time)