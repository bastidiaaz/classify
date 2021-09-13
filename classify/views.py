from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
from scipy.signal import resample
from scipy.interpolate import interp1d
import mne
import numpy as np
# from django.http import JsonResponse
# from django.core.files.base import ContentFile
# from django.conf import settings
# from tensorflow.python.keras.backend import set_session
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.applications.imagenet_utils import decode_predictions
# import matplotlib.pyplot as plt
# import numpy as np
# from keras.applications import vgg16
# import datetime
# import traceback

def ResampleLinear1D(original, targetLen):
    original = np.array(original, dtype=np.float)
    index_arr = np.linspace(0, len(original)-1, num=targetLen, dtype=np.float)
    index_floor = np.array(index_arr, dtype=np.int) #Round down
    index_ceil = index_floor + 1
    index_rem = index_arr - index_floor #Remain

    val1 = original[index_floor]
    val2 = original[index_ceil % len(original)]
    interp = val1 * (1.0-index_rem) + val2 * index_rem
    assert(len(interp) == targetLen)
    return interp

def index(request):
    if  request.method == "POST":
        # Save file
        f=request.FILES['sentFile'] # here you get the files needed
        response = {}
        file_name = "data.edf"
        file_name_2 = default_storage.save(file_name, f)
        file_url = default_storage.url(file_name_2)

        data = mne.io.read_raw_edf('D:/Proyectos/classify' + file_url)
        raw_data = data.get_data()

        signal_labels = data.ch_names
        n = 19
        # Se crea matriz para ingresar valores del registro. Nro de canales actual = 19.
        largo = raw_data.shape[1]
        freq = data.info['sfreq']
        freq_objetivo = 250

        factor = freq_objetivo/freq
        largo = largo * factor
        largo = int(largo)
        eeg = np.zeros((n, largo))
        largo_ventana = 4  # largo en segundos
        terms = largo_ventana * freq_objetivo

        # Asignación
        for i in np.arange(n):
            eeg[i, :] = ResampleLinear1D(raw_data[i],largo)

        print(eeg)

        return render(request,'homepage.html')
        # return render(request,'templates/homepage.html',response)
    else:
        return render(request,'homepage.html')
