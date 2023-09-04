import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.sequential import PARSynthesizer

patient_record = pd.read_csv('/content/sample_data/TrainDead.csv')
# patient_record = pd.read_csv('/content/sample_data/TrainSurvived.csv')

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=patient_record)
metadata.update_column(
    column_name='ID',
    sdtype='id')
metadata.set_sequence_key(column_name='ID')

synthesizer = PARSynthesizer(
    metadata,
    context_columns=['Age','Gender','Alcoholic','Smoke','FHCD','TriageScore','Outcome'],
    epochs=90,
    verbose=True,
    )

synthesizer.fit(patient_record)

#
synthetic_data = synthesizer.sample(num_sequences=74)
synthetic_data.to_csv("/content/sample_data/FS74.csv")