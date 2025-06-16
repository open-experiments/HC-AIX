#!/usr/bin/env python3
"""
Clinical Data Reader for Lokman-v2
Reads and translates the clinical Excel file to understand proper medical labels
"""

import pandas as pd
from pathlib import Path
import json

def read_clinical_excel(excel_path):
    """Read the Turkish clinical Excel file and translate to English"""
    
    print("📋 Reading Clinical Data from Excel File")
    print("=" * 50)
    
    try:
        # Read the Excel file with proper header row (found at row 5)
        df = pd.read_excel(excel_path, sheet_name='Sayfa1', skiprows=5, header=0)
        
        print(f"📊 Found {len(df)} patient records")
        print(f"📊 Columns: {list(df.columns)}")
        print()
        
        # Display first few rows to understand structure
        print("🔍 Sample Data (first 3 rows):")
        print("-" * 30)
        for idx, row in df.head(3).iterrows():
            print(f"Patient {idx + 1}:")
            for col, value in row.items():
                print(f"   {col}: {value}")
            print()
        
        # Analyze the data structure
        print("📈 Data Analysis:")
        print("-" * 20)
        
        # Check for common medical terms and translate
        turkish_translations = {
            'HASTA_ADI': 'PATIENT_NAME',
            'HASTA_SOYADI': 'PATIENT_SURNAME',
            'YAS': 'AGE',
            'ER': 'ER_STATUS',  # Estrogen Receptor
            'PR': 'PR_STATUS',  # Progesterone Receptor
            'brca 1': 'BRCA1',  # BRCA1 gene mutation
            'brca2': 'BRCA2',   # BRCA2 gene mutation
            'CerbB2': 'HER2',   # HER2/neu receptor
            'pik3ca mutasyonu': 'PIK3CA_MUTATION',
            'Ki-67': 'KI67_PROLIFERATION_INDEX',
            'MEME': 'BREAST',
            'MALIGN': 'MALIGNANT',
            'BENİGN': 'BENIGN',
            'NORMAL': 'NORMAL',
            'TÜMÖR': 'TUMOR',
            'TUMOR': 'TUMOR',
            'KANSER': 'CANCER',
            'SAĞLIKLI': 'HEALTHY',
            'TANISI': 'DIAGNOSIS',
            'BULGULAR': 'FINDINGS',
            'RAPOR': 'REPORT',
            'YORUM': 'COMMENT',
            'DURUM': 'STATUS',
            'SONUÇ': 'RESULT',
            'TIP': 'TYPE',
            'BÖLGE': 'REGION',
            'BOYUT': 'SIZE',
            'LOKASYON': 'LOCATION',
            'POZİTİF': 'POSITIVE',
            'NEGATİF': 'NEGATIVE',
            'YÜKSEK': 'HIGH',
            'DÜŞÜK': 'LOW',
            'ORTA': 'MODERATE'
        }
        
        # Try to identify key columns
        key_columns = []
        diagnosis_columns = []
        
        for col in df.columns:
            col_upper = str(col).upper()
            
            # Look for patient identification columns
            if any(term in col_upper for term in ['HASTA', 'ADI', 'İSİM', 'NAME']):
                key_columns.append(col)
                print(f"   🔍 Patient ID column: {col}")
            
            # Look for diagnosis/label columns
            if any(term in col_upper for term in ['TANISI', 'BULGULAR', 'SONUÇ', 'DURUM', 'TIP']):
                diagnosis_columns.append(col)
                print(f"   🏷️  Diagnosis column: {col}")
        
        print()
        
        # Analyze unique values in potential diagnosis columns
        if diagnosis_columns:
            print("🏥 Medical Diagnoses Found:")
            print("-" * 25)
            
            for col in diagnosis_columns:
                unique_values = df[col].dropna().unique()
                print(f"   Column '{col}':")
                
                for value in unique_values:
                    # Translate Turkish terms
                    translated = str(value).upper()
                    for turkish, english in turkish_translations.items():
                        if turkish in translated:
                            translated = translated.replace(turkish, english)
                    
                    print(f"      • {value} → {translated}")
                print()
        
        # Look for age information
        age_columns = [col for col in df.columns if 'YAŞ' in str(col).upper() or 'AGE' in str(col).upper()]
        if age_columns:
            print("👥 Age Information:")
            for col in age_columns:
                ages = df[col].dropna()
                if len(ages) > 0:
                    print(f"   Age range: {ages.min()} - {ages.max()} years")
                    print(f"   Mean age: {ages.mean():.1f} years")
        
        print()
        
        # Create a summary of the clinical data structure
        clinical_summary = {
            'total_patients': len(df),
            'columns': list(df.columns),
            'key_columns': key_columns,
            'diagnosis_columns': diagnosis_columns,
            'age_columns': age_columns,
            'sample_data': df.head(3).to_dict('records') if len(df) > 0 else []
        }
        
        return df, clinical_summary
        
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        return None, None

def map_clinical_to_labels(clinical_df):
    """Map clinical diagnoses to numerical labels for training"""
    
    if clinical_df is None:
        return None
    
    print("🔄 Mapping Clinical Diagnoses to Training Labels")
    print("=" * 50)
    
    # Define mapping from clinical biomarkers to training labels
    # Based on standard breast cancer prognostic factors
    clinical_to_label = {
        # Normal/Healthy cases - no positive markers
        'NORMAL': 0,
        'SAĞLIKLI': 0,
        'HEALTHY': 0,
        'NEGATİF': 0,  # All negative markers
        
        # Benign cases - ER/PR positive but low risk
        'BENİGN': 1,
        'BENIGN': 1,
        'İYİ HUYLU': 1,
        'ER_POSITIVE_LOW_RISK': 1,
        
        # Malignant cases - high-risk markers
        'MALIGN': 2,
        'MALIGNANT': 2,
        'KÖTÜ HUYLU': 2,
        'KANSER': 2,
        'CANCER': 2,
        'HER2_POSITIVE': 2,
        'TRIPLE_NEGATIVE': 2,
        
        # Tumor/High aggressive - multiple positive markers
        'TÜMÖR': 3,
        'TUMOR': 3,
        'KITLE': 3,
        'MASS': 3,
        'BRCA_POSITIVE': 3,
        'HIGH_KI67': 3
    }
    
    # Try to find diagnosis in any text column
    label_mapping = {}
    
    for idx, row in clinical_df.iterrows():
        patient_label = 0  # Default to normal
        patient_info = []
        
        # Extract patient identifier first
        patient_id = None
        if 'HASTA_ADI' in row and pd.notna(row['HASTA_ADI']):
            patient_id = str(row['HASTA_ADI']).strip()
        
        if not patient_id:
            continue  # Skip rows without patient ID
        
        # Analyze clinical biomarkers for proper classification
        risk_score = 0
        biomarkers = {}
        
        # Check all available biomarkers
        for col, value in row.items():
            if pd.notna(value):
                col_str = str(col).upper()
                value_str = str(value).upper()
                patient_info.append(f"{col}: {value}")
                
                # Age factor
                if 'YAS' in col_str:
                    try:
                        age = int(float(value))
                        biomarkers['age'] = age
                        if age > 50:
                            risk_score += 1  # Higher age = higher risk
                    except:
                        pass
                
                # ER (Estrogen Receptor) status
                elif 'ER' in col_str:
                    biomarkers['ER'] = value_str
                    if 'POZİTİF' in value_str or 'POSITIVE' in value_str or '+' in value_str:
                        risk_score += 1
                    elif 'NEGATİF' in value_str or 'NEGATIVE' in value_str or '-' in value_str:
                        risk_score += 2  # ER negative is higher risk
                
                # PR (Progesterone Receptor) status  
                elif 'PR' in col_str:
                    biomarkers['PR'] = value_str
                    if 'POZİTİF' in value_str or 'POSITIVE' in value_str or '+' in value_str:
                        risk_score += 1
                    elif 'NEGATİF' in value_str or 'NEGATIVE' in value_str or '-' in value_str:
                        risk_score += 2
                
                # HER2 status (CerbB2)
                elif 'CERBB2' in col_str or 'HER2' in col_str:
                    biomarkers['HER2'] = value_str
                    if 'POZİTİF' in value_str or 'POSITIVE' in value_str or '+' in value_str:
                        risk_score += 3  # HER2 positive is high risk
                
                # BRCA mutations
                elif 'BRCA' in col_str:
                    biomarkers['BRCA'] = value_str
                    if 'POZİTİF' in value_str or 'POSITIVE' in value_str or 'MUTASYON' in value_str:
                        risk_score += 4  # BRCA mutation is very high risk
                
                # Ki-67 proliferation index
                elif 'KI-67' in col_str or 'KI67' in col_str:
                    biomarkers['KI67'] = value_str
                    try:
                        # Extract percentage if present
                        import re
                        ki67_match = re.search(r'(\d+)', value_str)
                        if ki67_match:
                            ki67_val = int(ki67_match.group(1))
                            biomarkers['KI67_value'] = ki67_val
                            if ki67_val > 20:
                                risk_score += 3  # High Ki-67 indicates aggressive tumor
                            elif ki67_val > 10:
                                risk_score += 1
                    except:
                        pass
                
                # PIK3CA mutation
                elif 'PIK3CA' in col_str:
                    biomarkers['PIK3CA'] = value_str
                    if 'MUTASYON' in value_str or 'POSITIVE' in value_str:
                        risk_score += 2
        
        # Determine final label based on risk score and biomarker profile
        if risk_score == 0:
            patient_label = 0  # Normal/No significant risk factors
        elif risk_score <= 2:
            patient_label = 1  # Benign/Low risk
        elif risk_score <= 5:
            patient_label = 2  # Malignant/High risk
        else:
            patient_label = 3  # Tumor/Very high risk
        
        # Override based on specific biomarker combinations
        if biomarkers.get('BRCA') and 'POZİTİF' in str(biomarkers['BRCA']):
            patient_label = 3  # BRCA positive = highest risk
        elif biomarkers.get('HER2') and 'POZİTİF' in str(biomarkers['HER2']):
            # Check for triple negative (ER-, PR-, HER2+)
            er_neg = biomarkers.get('ER') and 'NEGATİF' in str(biomarkers['ER'])
            pr_neg = biomarkers.get('PR') and 'NEGATİF' in str(biomarkers['PR'])
            if er_neg and pr_neg:
                patient_label = 3  # Triple negative-like = highest risk
            else:
                patient_label = 2  # HER2 positive = high risk
        
        label_mapping[patient_id] = {
            'label': patient_label,
            'info': patient_info,
            'biomarkers': biomarkers,
            'risk_score': risk_score
        }
    
    print(f"📋 Mapped {len(label_mapping)} patients:")
    
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for patient, data in label_mapping.items():
        label = data['label']
        label_counts[label] += 1
        
        label_names = {0: 'Normal', 1: 'Benign', 2: 'Malignant', 3: 'Tumor'}
        print(f"   {patient}: Label {label} ({label_names[label]})")
    
    print(f"\n📊 Label Distribution:")
    for label, count in label_counts.items():
        label_names = {0: 'Normal', 1: 'Benign', 2: 'Malignant', 3: 'Tumor'}
        percentage = (count / len(label_mapping)) * 100 if len(label_mapping) > 0 else 0
        print(f"   {label} ({label_names[label]}): {count} patients ({percentage:.1f}%)")
    
    return label_mapping

def main():
    excel_path = "data/data-original/MEME HASTA EXCEL LİSTESİ.xlsx"
    
    if not Path(excel_path).exists():
        print(f"❌ Excel file not found: {excel_path}")
        return
    
    # Read clinical data
    clinical_df, summary = read_clinical_excel(excel_path)
    
    if clinical_df is not None:
        # Map to training labels
        label_mapping = map_clinical_to_labels(clinical_df)
        
        # Save the mapping for use in data preparation
        if label_mapping:
            output_file = "data/clinical_labels.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(label_mapping, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Clinical label mapping saved to: {output_file}")
            print("   This file can be used to properly label the training data")
    
    print("\n✅ Clinical data analysis complete!")

if __name__ == "__main__":
    main()