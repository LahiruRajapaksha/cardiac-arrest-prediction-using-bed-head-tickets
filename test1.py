import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.sequential import PARSynthesizer

patient_record = pd.read_csv('/content/sample_data/Train_Survived.csv')

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=patient_record)
metadata.update_column(
    column_name='ID',
    sdtype='id')
metadata.set_sequence_key(column_name='ID')
# print(metadata)#
# sequence = patient_record[patient_record['ID'] == 1]
# sequence

#
# patient_record['ID'].unique()


synthesizer = PARSynthesizer(
    metadata,
    context_columns=['Age',	'Gender', 'Alcoholic','FHCD','Smoke','TriageScore','Outcome'],
    epochs=2,
    verbose=True,
    )


synthesizer.fit(patient_record)

#
synthetic_data = synthesizer.sample(num_sequences=2)
print(synthetic_data)