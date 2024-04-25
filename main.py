import argparse
import pandas as pd
from helpers.eda_and_preprocessing import preprocess_data
from helpers import nhits,nbeats,tft
from pytorch_forecasting.metrics import MAE, SMAPE, RMSE
from pytorch_forecasting import Baseline
import torch

import warnings

warnings.filterwarnings("ignore")

def calculate_metrics(model,test_dataloader):

    actuals = torch.cat([y[0] for x, y in iter(test_dataloader)])
    predictions = model.predict(test_dataloader)
    smape = SMAPE()(predictions, actuals)
    mae = MAE()(predictions, actuals)
    rmse = RMSE()(predictions, actuals)
    metrics_list=[round(mae.item(),2),round(rmse.item(),2),round(smape.item(),2)]
    return metrics_list

    


def evaluation_metrics(nhit_model,nbeats_model,tft_model,val_dataloader):

    #Calculating Baseline Metrics
    metrics_list_baseline=calculate_metrics(Baseline(),val_dataloader)
    if nhit_model!='':
        metrics_list_nhits=calculate_metrics(nhit_model,val_dataloader)
    else:
        metrics_list_nhits=['','','']

    if nbeats_model!='':
        metrics_list_nbeats=calculate_metrics(nbeats_model,val_dataloader)
    else:
        metrics_list_nbeats=['','','']

    if tft_model!='':
        metrics_list_tft=calculate_metrics(tft_model,val_dataloader)
    else:
        metrics_list_tft=['','','']

    # Define index and column names
    indices = ['Baseline', 'TFT', 'NHITS', 'NBEATS']
    columns = ['MAE', 'RMSE', 'SMAPE']

    # Create an empty DataFrame
    df = pd.DataFrame(index=indices, columns=columns)

    df.loc['Baseline'] = metrics_list_baseline
    df.loc['TFT'] = metrics_list_tft
    df.loc['NHITS'] = metrics_list_nhits
    df.loc['NBEATS'] = metrics_list_nbeats

    return df


def main():

    parser = argparse.ArgumentParser(description='Empirical Analysis of Rossman Data Using NBeats,NHits and TFT ')
    parser.add_argument('--model', type=str, help='The model you want to run')
    parser.add_argument('--no_of_epochs', type=int, help='No of epochs you want to run the algorithm')
    parser.add_argument('--all', action='store_true', help='Include this flag if you want to run all 3 models,Otherwise only the specified model will run.')
    args = parser.parse_args()

    #No of columns we want to consider for training
    columns_to_select = ['Store','Date','Sales','Customers','Promo','StateHoliday','StoreType','IsPromoMonth','Day','Month','DayOfWeek','CompetitionOpen']
    max_encoder_length=180
    max_prediction_length=45
    target_column="Sales"
    tft_model=''
    nbeats_model=''
    nhits_model=''
    if args.all :
        #Preprocessing the data with Exploratory Data Analysis
        print("Step 1: PREPROCESSING AND EDA...")
        preprocessed_data=preprocess_data("datasets/raw/train.csv","datasets/raw/store.csv",columns_to_select=columns_to_select)
        #Training TFT
        print("Step 2: TRAINING TFT...")
        train_dataloader,val_dataloader,test_dataloader,train_tft=tft.create_timeseries_dataset(max_encoder_length=max_encoder_length,max_prediction_length=max_prediction_length,preprocessed_data=preprocessed_data,target_column=target_column)
        tft_model=tft.train_tft(train_dataloader,val_dataloader,train_tft,max_prediction_length=max_prediction_length,no_of_epochs=args.no_of_epochs)
        tft.generate_forecasts(tft_model,test_dataloader)

        print("Step 3: TRAINING NBeats...")
        train_dataloader,val_dataloader,test_dataloader,train_nbeats=nbeats.create_timeseries_dataset(max_encoder_length=max_encoder_length,max_prediction_length=max_prediction_length,preprocessed_data=preprocessed_data,target_column=target_column)
        nbeats_model=nbeats.train_nbeats(train_dataloader,val_dataloader,train_nbeats,max_prediction_length=45,no_of_epochs=args.no_of_epochs)
        nbeats.generate_forecasts(nbeats_model,test_dataloader)

        print("Step 4: TRAINING NHits...")
        train_dataloader,val_dataloader,test_dataloader,train_nhits=nhits.create_timeseries_dataset(max_encoder_length=max_encoder_length,max_prediction_length=max_prediction_length,preprocessed_data=preprocessed_data,target_column=target_column)
        nhits_model=nhits.train_nhits(train_dataloader,val_dataloader,train_nhits,max_prediction_length=45,no_of_epochs=args.no_of_epochs)
        nhits.generate_forecasts(nhits_model,test_dataloader)

        #Evaluating All models
        print("Step 5: EVALUATING ALL MODELS...")

        df=evaluation_metrics(nhits_model,nbeats_model,tft_model,test_dataloader)
        df.to_csv("results/eval_metrics_for_comparison_all.csv")

        print("Evaluation Metrics.....")
        print(df)
 
    else:
        if args.model:
            print("Step 1: PREPROCESSING AND EDA...")
            #Preprocessing the data with Exploratory Data Analysis
            preprocessed_data=preprocess_data("datasets/raw/train.csv","datasets/raw/store.csv",columns_to_select=columns_to_select)

            if args.model=='nbeats':
                print("Step 2: TRAINING NBeats...")
                train_dataloader,val_dataloader,test_dataloader,train_nbeats=nbeats.create_timeseries_dataset(max_encoder_length=max_encoder_length,max_prediction_length=max_prediction_length,preprocessed_data=preprocessed_data,target_column=target_column)
                nbeats_model=nbeats.train_nbeats(train_dataloader,val_dataloader,train_nbeats,max_prediction_length=45,no_of_epochs=args.no_of_epochs)
                nbeats.generate_forecasts(nbeats_model,test_dataloader)

                #Evaluating NBeats
                print("Step 3: EVALUATING NBeats...")

                df=evaluation_metrics(nhits_model,nbeats_model,tft_model,test_dataloader)
                df.to_csv("results/eval_metrics_for_comparison_nbeats.csv")

                print("Evaluation Metrics.....")
                print(df)
            elif args.model=='nhits':

                print("Step 2: TRAINING NHits...")
                train_dataloader,val_dataloader,test_dataloader,train_nhits=nhits.create_timeseries_dataset(max_encoder_length=max_encoder_length,max_prediction_length=max_prediction_length,preprocessed_data=preprocessed_data,target_column=target_column)
                nhits_model=nhits.train_nhits(train_dataloader,val_dataloader,train_nhits,max_prediction_length=45,no_of_epochs=args.no_of_epochs)
                nhits.generate_forecasts(nhits_model,test_dataloader)

                #Evaluating NBeats
                print("Step 3: EVALUATING NHits...")

                df=evaluation_metrics(nhits_model,nbeats_model,tft_model,test_dataloader)
                df.to_csv("results/eval_metrics_for_comparison_nhits.csv")

                print("Evaluation Metrics.....")
                print(df)
            elif args.model=='tft':
                #Training TFT
                print("Step 2: TRAINING TFT...")
                train_dataloader,val_dataloader,test_dataloader,train_tft=tft.create_timeseries_dataset(max_encoder_length=max_encoder_length,max_prediction_length=max_prediction_length,preprocessed_data=preprocessed_data,target_column=target_column)
                tft_model=tft.train_tft(train_dataloader,val_dataloader,train_tft,max_prediction_length=max_prediction_length,no_of_epochs=args.no_of_epochs)
                tft.generate_forecasts(tft_model,test_dataloader)

                #Evaluating TFT
                print("Step 3: EVALUATING TFT...")
                df=evaluation_metrics(nhits_model,nbeats_model,tft_model,test_dataloader)
                df.to_csv("results/eval_metrics_for_comparison_tft.csv")
                print("Evaluation Metrics.....")
                print(df)
        else:
            print("Please provide a valid model name!")




if __name__ == "__main__":
    main()