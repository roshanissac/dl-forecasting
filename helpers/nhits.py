import matplotlib.pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import time
from pytorch_forecasting import  TimeSeriesDataSet,NHiTS
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import MQF2DistributionLoss


def create_timeseries_dataset(max_encoder_length,max_prediction_length,preprocessed_data,target_column='Sales'):

    print("Creating timeseries data and dataloaders for N-Hits...")

    # Start time
    start_time = time.time()
    total_length=max_prediction_length*2
    training_cutoff = preprocessed_data["time_idx"].max() - total_length
    validation_cutoff = preprocessed_data["time_idx"].max() - max_prediction_length
    context_length = max_encoder_length
    prediction_length = max_prediction_length
    preprocessed_data[target_column] = preprocessed_data[target_column].astype(np.float32)

    training_nhit = TimeSeriesDataSet(
        preprocessed_data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=target_column,
        categorical_encoders={"Store": NaNLabelEncoder(add_nan=True).fit(preprocessed_data.Store)},
        group_ids=["Store"],
        time_varying_unknown_reals=[target_column],
        max_encoder_length=context_length,
        max_prediction_length=prediction_length,
        allow_missing_timesteps=True
        
    )

    validation = TimeSeriesDataSet.from_dataset(training_nhit,preprocessed_data[lambda x: x.time_idx<=validation_cutoff], min_prediction_idx=training_cutoff + 1)
    test = TimeSeriesDataSet.from_dataset(training_nhit, preprocessed_data, min_prediction_idx=validation_cutoff + 1)
    batch_size = 128
    train_dataloader = training_nhit.to_dataloader(train=True, batch_size=batch_size, num_workers=8)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=8)
    test_dataloader = test.to_dataloader(train=False, batch_size=batch_size, num_workers=8)

    # End time
    end_time = time.time()

    # Calculate execution time
    execution_time = end_time - start_time

    print(f"Creating timeseries data and dataloaders for N-Hits finished in {execution_time} seconds")

    return train_dataloader,val_dataloader,test_dataloader,training_nhit

def train_nhits(train_dataloader,val_dataloader,training_nhit,max_prediction_length,no_of_epochs=1):

    print("Training N-Hits model...")

    # Start time
    start_time = time.time()

    pl.seed_everything(42)

    net = NHiTS.from_dataset(
    training_nhit,
    learning_rate=1e-3,
    log_interval=10,
    weight_decay=1e-2,
    backcast_loss_ratio=1.0,
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    
    trainer = pl.Trainer(
        max_epochs=no_of_epochs,
        accelerator="cpu",
        enable_model_summary=True,
        gradient_clip_val=1.0,
        callbacks=[early_stop_callback],
        limit_train_batches=50,
      
    )

    trainer.fit(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = NHiTS.load_from_checkpoint(best_model_path)

    # End time
    end_time = time.time()

    # Calculate execution time
    execution_time = end_time - start_time

    print(f"Training N-Hits model finished in {execution_time} seconds")

    return best_model


def generate_forecasts(model,data):

    test_prediction_results = model.predict(data, mode="raw", return_x=True)

    print("Generating Forecasts plots for 5 stores...")

    for idx in range(5):  # plot 5 examples
        fig, ax = plt.subplots(figsize=(23,5))
        model.plot_prediction(test_prediction_results.x, # network input
                              test_prediction_results.output, # network output
                              idx=idx,
                              add_loss_to_title=True,
                              ax=ax);
        plt.savefig(f'charts\/nhits\/forecast_store_{idx}.png')
        plt.close() 

    return ''






