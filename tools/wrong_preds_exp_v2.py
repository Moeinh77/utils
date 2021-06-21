import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def show_wrong_cases(predictions, true_classes, num_classes, images, binary_classes = True,
                     show_count_plot = True, most_confused = True, k = 4):
    
    # if the preds are for a 2 class problem, use a threshold
    if binary_classes == True:
        def binary_(preds):
            predictions = np.zeros(len(preds))
            for p in range(len(preds)):
                if preds[p] >= 0.5:
                    predictions[p] = 1

                else:
                    predictions[p] = 0
            return predictions

        predictions_binary = binary_(predictions)
#         print(predictions)   

    # create lists to save the number of wrong preds for each class
    wrong_preds_inds = [list() for i in range(num_classes)]
    
    # find the index of samples that are predicted mistakenly
    for i in range(len(predictions)):       
        if binary_classes:
            if predictions_binary[i] != true_classes[i]:
                wrong_preds_inds[true_classes[i]].append(i) # add the index i to the list related to its true class
                
        else:
            if np.argmax(predictions[i]) != np.argmax(true_classes[i]):
                wrong_preds_inds[np.argmax(true_classes[i])].append(i)
            
#                 print(predictions[i], true_classes[i])
                
    # show the number of wrong preds in each class on a bar plot
    if show_count_plot:
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_axes([0,0,1,1])
        x = [i for i in range(num_classes)]
        y = [len(w) for w in wrong_preds_inds]
        ax.bar(x, y)
        ax.set_title('false predicitons in each class')
        plt.show()
    
    # show the samples with the highest prediction loss
    ''' make it so that only wrong preds show '''
    if most_confused: 
        if binary_classes:
            LF = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        else:
            LF = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        losses = LF(true_classes, predictions).numpy()
#         print(binary_classes)
#         print(losses)
        
        highest_loss_ind = []
        
        wrong_preds_inds_all = [i for j in wrong_preds_inds for i in j]
        for loss_idx in range(len(losses)):
            if losses.argsort()[-loss_idx] in wrong_preds_inds_all:
                highest_loss_ind.append(losses.argsort()[-loss_idx])
                if len(highest_loss_ind) == k:
                    break
#         highest_loss_ind = losses.argsort()[-k:] # np.argmax(losses)
        
        losses.sort()
        highest_losses = losses[-k:] # np.max(losses)
        
        f, ax = plt.subplots(k//2, k//2, gridspec_kw={'hspace': 1, 'wspace': 1})
        
        ix_img= 0
        for i in range(0, k//2):
            for j in range(0, k//2):
                ax[i,j].imshow(images[highest_loss_ind[ix_img]])
                
                if binary_classes:
                    true_cls = true_classes[highest_loss_ind[ix_img]]
                    predicted_cls = predictions_binary[highest_loss_ind[ix_img]]
                else:
                    true_cls = np.argmax(true_classes[highest_loss_ind[ix_img]])
                    predicted_cls = np.argmax(predictions[highest_loss_ind[ix_img]])
                    
                ax[i,j].set_title('true class: '+str(true_cls)
                  +' predicted: '+ str(predicted_cls) +
                   '\n idx: '+ str(highest_loss_ind[ix_img]))

                ax[i,j].axis('off')
                ax[i,j].set_aspect('auto')
                ix_img += 1
                
        plt.show()
#     return wrong_preds_inds