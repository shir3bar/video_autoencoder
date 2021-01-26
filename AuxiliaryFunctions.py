# A set of functions to help with visualizations and other household chores
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import cv2
import os
import io


def get_plottable_frame(frame):
    """ Get a tensor of shape CxHxW and transform it into a numpy array of HxWxC"""
    frame = frame.numpy()
    frame = np.transpose(frame, (1, 2, 0))
    # if num channels == 1 get rid of the extra dimension:
    frame = np.squeeze(frame)
    return frame


def fig_to_img(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def one_frame_heatmap(input_frame,output_frame):
    if torch.is_tensor(input_frame):
        input_frame = input_frame.cpu().numpy()
    if torch.is_tensor(output_frame):
        output_frame = output_frame.detach().cpu().numpy()
    heat_map = np.sqrt((output_frame - input_frame)**2)
    return np.squeeze(heat_map)


def heat_video(input_vids, output_vids, directory,filename, fps=30, idx=0,batch_size=4):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    num_vids, color_channel, num_frames, h, w = input_vids.shape
    print(num_frames)
    try:
        os.mkdir(directory)
    except:
        pass
    for vid_num in range(num_vids):
        movie_path = directory+f'heatmap_{filename}.avi'#{batch_size*idx+vid_num}.avi'
        print(movie_path)
        vid_writer = cv2.VideoWriter(movie_path, fourcc, fps, (1080, 720), True)
        for frame_num in range(num_frames):
            input_frame = input_vids[vid_num, :, frame_num, :, :]
            output_frame = output_vids[vid_num, :, frame_num, :, :]
            heatmap = one_frame_heatmap(input_frame, output_frame)
            fig = plt.figure()
            sns.heatmap(heatmap, vmin=0, vmax=1)
            plt.axis('off')
            img = fig_to_img(fig)
            vid_writer.write(img)
            plt.close(fig)
        vid_writer.release()


def show_batch(clips_batch):
    """Show image with landmarks for a batch of samples."""
    if clips_batch.device != "cpu":
        clips_batch = clips_batch.cpu()
    batch_size = clips_batch.shape[0]
    clip_size = clips_batch.shape[2]
    for i in range(batch_size):
        plt.subplot(1,batch_size,i+1)
        frame = clips_batch[i, :, 1, :, :]  # Clip number, color channel, frame number, height, width
        frame = get_plottable_frame(frame)
        plt.axis('off')
        plt.ioff()
        plt.imshow(frame, cmap='gray')


def split_clip(load_path,save_path,num_frames,fps=30):
    cap=cv2.VideoCapture(load_path)
    fourcc=cv2.VideoWriter_fourcc(*"MJPG")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    counter=1
    cont=True
    while cont:
        movie_path = save_path + os.path.basename(load_path).split('.')[0] +\
             '_segment_' + str(counter) + '.avi'
        vid_writer = cv2.VideoWriter(movie_path, fourcc, fps, (H,W), True)
        for i in range(num_frames):
            ret, frame = cap.read()
            if ret:
                vid_writer.write(np.uint8(frame))
            else:
                cont = False
                break
        vid_writer.release()
        if i < num_frames-1:
            os.remove(movie_path)
        counter += 1


# Training auxiliary functions:

def save_checkpoint(model,optimizer,epoch,train_loss,scheduler_state_dict=None,
                    val_loss=None,directory='/content/drive/My Drive/PhD/Swim_samples',name='model'):
    path=os.path.join(directory,f'{name}_epoch{epoch}.pt')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict':scheduler_state_dict,
            'train_loss': train_loss,
            'val_loss': val_loss,
            }, path)


def save_recon(reconstruction,model_name,epoch,directory):
    outputs1 = reconstruction.detach().cpu().numpy()
    output_frame = outputs1[0, 0, 2, :, :]
    fig=plt.figure()
    plt.imshow(output_frame, cmap='gray')
    plt.axis('off')
    plt.title(f'{model_name}_epoch{epoch}')
    img=fig_to_img(fig)
    plt.close(fig)
    filename=f'{model_name}_epoch{epoch}.jpg'
    filepath = os.path.join(directory, filename)
    cv2.imwrite(filepath, img)

