import torch, os, pickle, time, copy, sys



def mkdir_model(base_dir, name_model, counter):
  """
  Making a directory for the model dump.
  """
  try:
    d = "{}/{}".format(base_dir,name_model)
    os.mkdir(d)
  except FileExistsError:
    counter += 1
    mkdir_model(base_dir, str(name_model) + "_" + str(counter), counter)

def save_history(history, filename):
  """
  Save the history in the file.
  """
  if os.path.isfile(filename):
    os.remove(filename)
  file_handler = open(filename + ".pkl", "wb")
  pickle.dump(history, file_handler)
  file_handler.close()


def load_history(filename):
  """
  Load the history from the file.
  """
  file_handler = open(filename + ".pkl", "rb")
  output = pickle.load(file_handler)
  file_handler.close()
  return output # 가중치 파일 인식?


def save_checkpoint(model, optimizer, epoch, loss, accuracy, filename='checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filename)