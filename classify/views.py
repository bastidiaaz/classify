from django.shortcuts import render
import pyedflib
import numpy as np

def index(request):
    if  request.method == "POST":
        # file=request.FILES['sentFile']
        # response = {}
        # f = pyedflib.EdfReader(file)
        #
        # print(f)
        # print(f.getSignalLabels())

        # # Archivo a trabajar
        # file_path = root + rec
        # # Lectura
        # # Numero de canales
        # n = 19
        # # n = f.signals_in_file
        # signal_labels = f.getSignalLabels()
        # # Se crea matriz para ingresar valores del registro. Nro de canales actual = 19.
        # largo = f.getNSamples()[0]
        # freq = f.getSampleFrequency(0)
        # freq_objetivo = 250
        #
        # factor = freq_objetivo/freq
        # largo = largo * factor
        # largo = int(largo)
        # eeg = np.zeros((n, largo))
        # largo_ventana = 4  # largo en segundos
        # terms = largo_ventana * freq_objetivo

        # Asignación
        for i in np.arange(n):
            eeg[i, :] = ResampleLinear1D(f.readSignal(i),largo)

        # file_name = "pic.jpg"
        # file_name_2 = default_storage.save(file_name, f)
        # file_url = default_storage.url(file_name_2)
        # original = load_img(file_url, target_size=(224, 224))
        # numpy_image = img_to_array(original)
        #
        #
        # image_batch = np.expand_dims(numpy_image, axis=0)
        # # prepare the image for the VGG model
        # processed_image = vgg16.preprocess_input(image_batch.copy())
        #
        # # get the predicted probabilities for each class
        # with settings.GRAPH1.as_default():
        #     set_session(settings.SESS)
        #     predictions=settings.VGG_MODEL.predict(processed_image)
        #
        # label = decode_predictions(predictions)
        # label = list(label)[0]
        # response['name'] = str(label)
        return render(request,'homepage.html')
        # return render(request,'templates/homepage.html',response)
    else:
        return render(request,'homepage.html')
