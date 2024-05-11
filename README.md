
<h1>Low Cost LTE Positioning with nrf9160 Board</h1>

The aim of this project is to obtain a low cost positioning based on LTE. The signal strength and neighbor base station data obtained through nrf9160 board is processed with algorithms
such as multilateration, LSTM, random forrest to localize the device. 

<h2>Installation</h2>

To install the libraries in project root folder enter following command:

```
conda create --name nrf9160 --file requirements.txt
```

<h2>Run application</h2>

To run the application run the following command:

```
streamlit run .\streamlit_app.py
```

<h2>Capture data from nrf9160 board</h2>
To capture measurements run script on project root folder:

```
python3 capture_measurement.py
```
