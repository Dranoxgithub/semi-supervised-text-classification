from tensorboard.backend.event_processing import event_accumulator
import os
dir_name = 'Nov15_22-23-30_zavlanos-gpu dataset=dbpedia epochs=50 CE=True AT=True VAT=True EM=True'
ea = event_accumulator.EventAccumulator(os.path.join('/home/yanpeng/our_ssl/runs/', dir_name))

ea.Reload()
val_event_list = ea.Scalars('validation accuracy')
val_acc_list = []
for event in val_event_list:
    val_acc_list.append(event.value)
print(dir_name)
print(val_acc_list)
