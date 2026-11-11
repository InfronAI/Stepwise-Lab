"""Compute class-balanced metrics from stored confusion matrices; no model rerun."""
from pathlib import Path
import json

from prm800k.io import write_json

for path in Path('experiments').glob('m1-*/**/*training.json'):
    data=json.loads(path.read_text())
    for entry in data.get('history',[]):
        c=entry['confusion_matrix']
        support=[sum(row) for row in c]
        recall=[c[i][i]/s if s else 0 for i,s in enumerate(support)]
        precision=[c[i][i]/max(1,sum(row[i] for row in c)) for i in range(len(c))]
        present=[i for i,s in enumerate(support) if s]
        entry['balanced_accuracy']=sum(recall[i] for i in present)/len(present)
        entry['macro_f1']=sum(2*precision[i]*recall[i]/max(1e-12,precision[i]+recall[i]) for i in present)/len(present)
        entry['majority_class_baseline']=max(support)/sum(support)
    write_json(path,data)
