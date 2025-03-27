# HousingPricesPrediction

- Este proyecto se puede probar en [Google Colab](https://colab.research.google.com/) o en entorno local con **GPU**. Para ello:

1. Se pueden seguir las instrucciones del [notebook](https://github.com/alejandrorod-tajamar/HousingPricesPrediction/blob/main/Housing.ipynb) de una en una, aunque es recomendable simplemente revisarlas hasta llegar al punto _**5 Serialización del modelo**_.

2. En dicho punto, se debe ejecutar la tercera celda para cargar el modelo, que ya está exportado en la carpeta del proyecto.

3. Después, se pueden extraer datos aleatoriamente desde el _dataset_ para probarlos con la API del modelo.

4. También se puede ver la predicción para cada dato desde el propio _notebook_, junto con el porcentaje de error y la comparación del precio real contra el de la predicción. De esta manera, se pueden comparar estos resultados con los de la API para comprobar que son los mismos. La API se puede probar mediante línea de comandos o mediante [Postman](https://www.postman.com/). Para ejecutar la aplicación y el endpoint de la API:

```batch
python app.py
```

5. Enviar las peticiones al endpoint `http://127.0.0.1:5000/predict`
