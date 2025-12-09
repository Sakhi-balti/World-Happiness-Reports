
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData,PredictionPipeline

#------------Initialized Flask App --------->
application = Flask(__name__)

app = application

#------------Home Route-------------------->
@app.route('/')
def index():

    return render_template('home.html')
# ---------Predict Route--------------------->
@app.route('/predict', methods = ['GET', 'POST'])
def predic_datapoint():
    if request.method =='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            year = int(request.form.get('Year')),
            gdp = float(request.form.get('GDP')),
            social = float(request.form.get('SocialSupport')),
            life = float(request.form.get('LifeExpectancy')),
            freedom = float(request.form.get('Freedom')),
            generosity = float(request.form.get('Generosity')),
            corruption = float(request.form.get('Corruption'))
        )
        # call the dataframe function
        pred_df = data.get_data_frame()
        print( pred_df )

        prediction_pipeline =  PredictionPipeline()
        results = prediction_pipeline.predict(pred_df)
        return render_template('home.html', result = results[0])
    


# if __name__ == '__main__':
#     app.run(debug=True)