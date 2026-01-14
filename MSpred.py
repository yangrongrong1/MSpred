import os
import joblib
import numpy as np
import pandas as pd
import argparse
import itertools
import re
import sys
from typing import List, Dict, Optional, Any, Union
from Bio.SeqUtils import ProtParam, molecular_weight

# Set environment variables for custom module import
os.environ["PYTHONPATH"] = "/home/yangrongrong/PhD_study/Induced_tolerance_peptide/Induced/Induced_code"
sys.path.append("/home/yangrongrong/PhD_study/Induced_tolerance_peptide/Induced/Induced_code")
from feature import *  # Import all feature generation functions

# ========================== Initialize Models and Preprocessing Objects ==========================
def load_models():
    print("⚙️ Loading models and preprocessing objects...")
    MODEL_PATH = "/home/yangrongrong/PhD_study/Induced_tolerance_peptide/Induced/Induced_result/new/new_jicheng334.joblib"
    PREPROCESSING_OBJECTS_PATH = "/home/yangrongrong/PhD_study/Induced_tolerance_peptide/Induced/Induced_result/new/preprocessor.joblib"

    try:
        # Load preprocessing objects
        preprocessing_objects = joblib.load(PREPROCESSING_OBJECTS_PATH)
        selected_features = preprocessing_objects["selected_features"]
        selected_feature_names = preprocessing_objects["selected_feature_names"]
        selected_sets = preprocessing_objects["selected_feature_sets"]
        feature_selector = preprocessing_objects["feature_selector"]
        imputer = preprocessing_objects["imputer"]
        scaler = preprocessing_objects["scaler"]
        
        # Load model
        model = joblib.load(MODEL_PATH)
        print("✓ Successfully loaded models and preprocessing objects | Selected feature sets:", selected_sets)
        return model, feature_selector, imputer, scaler, selected_sets
    except Exception as e:
        print(f"⚠️ Failed to load models or preprocessing objects: {str(e)}")
        print("Please confirm the model and preprocessing object paths are correct, or retrain the model")
        exit(1)

# Load models and preprocessing objects globally
model, feature_selector, imputer, scaler, selected_sets = load_models()

# ====================== Physicochemical Properties Calculation Functions ========================
def calculate_physicochemical_properties(sequence: str) -> dict:
    """Calculate various physicochemical properties of peptide sequence"""
    properties = {}
    
    # Calculate basic properties using Biopython
    protein_analysis = ProtParam.ProteinAnalysis(sequence)
    
    # 1. Amphipathicity
    hydrophobic_aa = "ACFILMVWY"
    hydrophilic_aa = "DEHKNQRST"
    hydrophobic_count = sum(1 for aa in sequence if aa in hydrophobic_aa)
    hydrophilic_count = sum(1 for aa in sequence if aa in hydrophilic_aa)
    properties["amphipathicity"] = round((hydrophilic_count - hydrophobic_count) / len(sequence), 4) if sequence else 0
    
    # 2. Charge
    properties["charge"] = round(protein_analysis.charge_at_pH(7.0), 4)  # Charge at pH=7
    
    # 3. Hydropathicity
    kyte_doolittle = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    properties["hydropathicity"] = round(sum(kyte_doolittle.get(aa, 0) for aa in sequence) / len(sequence), 4)
    
    # 4. Hydrophilicity - Hopp-Woods scale
    hopp_woods = {
        'A': -0.5, 'R': 3.0, 'N': 0.2, 'D': 3.0, 'C': -1.0,
        'Q': 0.2, 'E': 3.0, 'G': 0.0, 'H': -0.5, 'I': -1.8,
        'L': -1.8, 'K': 3.0, 'M': -1.3, 'F': -2.5, 'P': 0.0,
        'S': 0.3, 'T': -0.4, 'W': -3.4, 'Y': -2.3, 'V': -1.5
    }
    properties["hydrophilicity"] = round(sum(hopp_woods.get(aa, 0) for aa in sequence) / len(sequence), 4)
    
    # 5. Hydrophobicity
    eisenberg = {
        'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
        'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
        'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
        'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
    }
    properties["hydrophobicity"] = round(sum(eisenberg.get(aa, 0) for aa in sequence) / len(sequence), 4)
    
    # 6. Molecular weight
    properties["molecular_weight"] = round(molecular_weight(sequence, seq_type="protein"), 2)
    
    # 7. Net Hydrogen
    donors = "HKRWY"
    acceptors = "DENQSTY"
    properties["net_hydrogen"] = round(
        sum(1 for aa in sequence if aa in donors) - 
        sum(1 for aa in sequence if aa in acceptors), 2
    )
    
    # 8. pI (isoelectric point)
    properties["pi"] = round(protein_analysis.isoelectric_point(), 4)
    
    # 9. Side bulk (side chain volume)
    bulkiness = {
        'A': 3, 'R': 14, 'N': 8, 'D': 8, 'C': 8,
        'Q': 11, 'E': 11, 'G': 1, 'H': 10, 'I': 9,
        'L': 9, 'K': 12, 'M': 11, 'F': 14, 'P': 7,
        'S': 6, 'T': 7, 'W': 21, 'Y': 14, 'V': 7
    }
    properties["side_bulk"] = round(sum(bulkiness.get(aa, 0) for aa in sequence) / len(sequence), 4)
    
    # 10. Steric hinderance
    steric_hinderance = {
        'A': 0.31, 'R': 1.01, 'N': 0.60, 'D': 0.60, 'C': 0.84,
        'Q': 0.78, 'E': 0.72, 'G': 0.00, 'H': 0.80, 'I': 1.25,
        'L': 1.22, 'K': 0.96, 'M': 0.96, 'F': 1.32, 'P': 0.72,
        'S': 0.49, 'T': 0.73, 'W': 1.65, 'Y': 1.30, 'V': 0.96
    }
    properties["steric_hinderance"] = round(sum(steric_hinderance.get(aa, 0) for aa in sequence) / len(sequence), 4)
    
    return properties

# ====================== Utility Functions ========================
def is_valid_sequence(seq: str) -> bool:
    """Verify if sequence contains only valid amino acids"""
    VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
    return set(seq).issubset(VALID_AA)

def generate_single_feature_set(feature_name: str, sequences: List[str]) -> pd.DataFrame:
    """Generate a single feature set"""
    func_name = f"{feature_name.lower()}_wp"
    try:
        func = globals().get(func_name)
        if not func or not callable(func):
            raise ValueError(f"Feature generation function not found: {func_name}")
        
        sequences_series = pd.Series(sequences)
        return func(sequences_series)
    except Exception as e:
        raise ValueError(f"Feature generation error ({feature_name}): {str(e)}")

def generate_and_process_features(sequences: List[str]) -> np.ndarray:
    """Generate and process all features"""
    feature_dfs = []
    for feature_set in selected_sets:
        df = generate_single_feature_set(feature_set, sequences)
        feature_dfs.append(df)
    
    merged_df = pd.concat(feature_dfs, axis=1)
    imputed_arr = imputer.transform(merged_df.values)
    scaled_arr = scaler.transform(imputed_arr)
    selected_arr = feature_selector.transform(scaled_arr)
    
    return selected_arr

def predict(sequences: List[str], probability_threshold: float = 0.7) -> tuple:
    """Predict peptide sequences in batch"""
    X = generate_and_process_features(sequences)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else predictions.astype(float)
    
    # Get probabilities and labels
    labels = ["MS inducer" if proba >= probability_threshold else "MS non-inducer" for proba in probabilities]
    
    # Get physicochemical properties
    properties = [calculate_physicochemical_properties(seq) for seq in sequences]
    
    return predictions, probabilities, labels, properties

def read_fasta_file(file_path: str) -> tuple:
    """Read FASTA file, return sequence IDs and sequences"""
    sequences = []
    ids = []
    current_seq = ""
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('>'):
                if current_seq:
                    sequences.append(current_seq)
                    ids.append(current_id)
                current_id = line[1:].split()[0]
                current_seq = ""
            else:
                current_seq += line
                
    if current_seq:
        sequences.append(current_seq)
        ids.append(current_id)
    
    return ids, sequences

def read_multisequence_file(file_path: str) -> list:
    """Read multi-sequence file, one sequence per line"""
    sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            seq = line.strip()
            if seq:
                sequences.append(seq.upper())
    return sequences

def save_results_to_csv(results: list, output_file: str):
    """Save results to CSV file"""
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"✓ Results saved to: {output_file}")
    except Exception as e:
        print(f"⚠️ Failed to save results: {str(e)}")

# ====================== Core Functionality ========================
def run_predict_mode(args):
    """Run predict mode"""
    print("\n🔮 Entering Predict Mode")
    print("-" * 50)
    
    sequences = []
    ids = []
    
    # Determine input source
    if args.sequence:
        sequences.append(args.sequence.upper())
    elif args.file:
        sequences = read_multisequence_file(args.file)
    elif args.fasta:
        ids, sequences = read_fasta_file(args.fasta)
    
    if not sequences:
        print("⚠️ Error: No valid input sequences provided")
        return
    
    # Validate sequences
    invalid_seqs = []
    for i, seq in enumerate(sequences):
        if not is_valid_sequence(seq):
            invalid_seqs.append((i+1, seq))
    
    if invalid_seqs:
        print("⚠️ Error: Invalid sequences detected:")
        for i, seq in invalid_seqs:
            invalid_aa = set(seq) - set("ACDEFGHIKLMNPQRSTVWY")
            print(f"  Sequence #{i}: {seq} - Invalid amino acids: {', '.join(invalid_aa)}")
        return
    
    # Execute prediction
    try:
        _, probabilities, labels, properties = predict(
            sequences, 
            probability_threshold=args.threshold
        )
        
        # Prepare results
        results = []
        high_risk_count = 0
        
        for i in range(len(sequences)):
            result = {
                "id": ids[i] if ids else None,
                "sequence": sequences[i],
                "probability": probabilities[i],
                "label": labels[i],
            }
            
            # Add physicochemical properties
            if args.properties:
                result.update(properties[i])
            
            results.append(result)
            
            if labels[i] == "MS inducer":
                high_risk_count += 1
        
        # Display results
        print("\n📊 Prediction Results:")
        print("=" * 80)
        print(f"🔹 Number of sequences: {len(sequences)}")
        print(f"🔴 High-risk peptides (MS inducer): {high_risk_count}")
        print(f"🟢 Safe peptides (MS non-inducer): {len(sequences) - high_risk_count}")
        
        # Print first few results
        display_count = min(args.limit, len(results))
        print(f"\nDisplaying first {display_count} results:")
        
        for i, res in enumerate(results[:display_count]):
            print(f"\n🔹 {'ID: ' + res['id'] + ', ' if res['id'] else ''}Sequence {i+1}: {res['sequence']}")
            print(f"   Predicted probability: {res['probability']:.4f}")
            print(f"   Predicted label: {res['label']}")
            
            if args.properties:
                print("   Physicochemical properties:")
                for prop in ['amphipathicity', 'charge', 'hydropathicity', 'hydrophilicity', 
                             'hydrophobicity', 'molecular_weight', 'net_hydrogen', 'pi', 
                             'side_bulk', 'steric_hinderance']:
                    if prop in res:
                        formatted_prop = prop.capitalize().replace('_', ' ')
                        print(f"     - {formatted_prop}: {res[prop]}")
        
        # Save results
        if args.output:
            save_results_to_csv(results, args.output)
    
    except Exception as e:
        print(f"⚠️ Prediction failed: {str(e)}")

def run_design_mode(args):
    """Run design mode (generate all single-point mutants)"""
    print("\n🧬 Entering Design Mode")
    print("-" * 50)
    
    sequence = args.sequence.upper()
    if not sequence:
        print("⚠️ Error: Sequence cannot be empty")
        return
        
    if not is_valid_sequence(sequence):
        invalid = set(sequence) - set("ACDEFGHIKLMNPQRSTVWY")
        print(f"⚠️ Error: Sequence contains invalid amino acids: {', '.join(invalid)}")
        return
    
    # Generate all mutants
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    
    mutants = []
    for i, orig_aa in enumerate(sequence):
        for new_aa in AMINO_ACIDS:
            if new_aa == orig_aa:
                continue
            mutant_seq = sequence[:i] + new_aa + sequence[i+1:]
            mutants.append({
                "position": i+1,
                "original_aa": orig_aa,
                "mutant_aa": new_aa,
                "mutant_sequence": mutant_seq
            })
    
    if not mutants:
        print("No mutants generated")
        return
    
    print(f"Generated {len(mutants)} mutants")
    
    # Predict all mutants
    try:
        mutant_seqs = [m["mutant_sequence"] for m in mutants]
        _, probabilities, labels, properties = predict(
            mutant_seqs, 
            probability_threshold=args.threshold
        )
        
        # Process results
        high_risk_mutations = []
        results = []
        
        for idx, mutant in enumerate(mutants):
            mutant["probability"] = probabilities[idx]
            mutant["label"] = labels[idx]
            
            if args.properties:
                mutant.update(properties[idx])
            
            results.append(mutant)
            
            if labels[idx] == "MS inducer":
                high_risk_mutations.append(mutant)
        
        # Display results
        print("\n📊 Mutant Design Results:")
        print("=" * 80)
        print(f"🔹 Original sequence: {sequence}")
        print(f"🔹 Risk threshold: {args.threshold}")
        print(f"🔴 High-risk mutations: {len(high_risk_mutations)}")
        print(f"🟢 Low-risk mutations: {len(mutants) - len(high_risk_mutations)}")
        
        # Display high-risk mutations
        if high_risk_mutations:
            print("\n🔴 High-risk mutants (MS inducer):")
            for i, m in enumerate(high_risk_mutations[:args.limit]):
                print(f"{i+1}. Position {m['position']}: {m['original_aa']}→{m['mutant_aa']}")
                print(f"   Mutant sequence: {m['mutant_sequence']}")
                print(f"   Predicted probability: {m['probability']:.4f}")
                
                if args.properties:
                    print("   Physicochemical properties:")
                    for prop in ['amphipathicity', 'charge', 'hydropathicity', 'hydrophilicity', 
                                'hydrophobicity', 'molecular_weight', 'net_hydrogen', 'pi', 
                                'side_bulk', 'steric_hinderance']:
                        if prop in m:
                            formatted_prop = prop.capitalize().replace('_', ' ')
                            print(f"     - {formatted_prop}: {m[prop]}")
                print("-" * 40)
        
        # Save results
        if args.output:
            save_results_to_csv(results, args.output)
    
    except Exception as e:
        print(f"⚠️ Mutant prediction failed: {str(e)}")

def run_scan_mode(args):
    """Run scan mode (sliding window analysis of protein sequence)"""
    print("\n🔍 Entering Scan Mode")
    print("-" * 50)
    
    # Read FASTA file
    try:
        ids, sequences = read_fasta_file(args.fasta)
        if not sequences:
            print("⚠️ Error: No valid sequences found in FASTA file")
            return
            
        sequence = sequences[0]
        protein_id = ids[0] if ids else "Unknown protein"
    except Exception as e:
        print(f"⚠️ Failed to read FASTA file: {str(e)}")
        return
    
    print(f"🔹 Protein ID: {protein_id}")
    print(f"🔹 Sequence length: {len(sequence)} aa")
    
    # Validate sequence
    if not is_valid_sequence(sequence):
        invalid = set(sequence) - set("ACDEFGHIKLMNPQRSTVWY")
        print(f"⚠️ Error: Sequence contains invalid amino acids: {', '.join(invalid)}")
        return
        
    # Window settings
    window_size = args.window
    if window_size < 4 or window_size > 50:
        print(f"⚠️ Warning: Window size should be between 4-50, using default value 9")
        window_size = 9
    
    # Perform scan
    try:
        peptide_windows = []
        positions = []
        
        # Generate all sliding window peptides
        for i in range(0, len(sequence) - window_size + 1):
            peptide = sequence[i:i+window_size]
            peptide_windows.append(peptide)
            positions.append((i+1, i+window_size))
        
        if not peptide_windows:
            print("No window peptides generated")
            return
            
        print(f"Generated {len(peptide_windows)} {window_size}-aa windows")
        print("🔄 Predicting window peptide risks...")
        
        # Predict all windows
        _, probabilities, labels, properties = predict(
            peptide_windows, 
            probability_threshold=args.threshold
        )
        
        # Process results
        hotspots = []
        results = []
        hotspot_count = 0
        
        for i, (start, end) in enumerate(positions):
            if labels[i] == "MS inducer":
                hotspot_count += 1
                hotspots.append({
                    "start": start,
                    "end": end,
                    "peptide": peptide_windows[i],
                    "probability": probabilities[i],
                    "overlap_count": 0  # Initialize overlap count
                })
            
            result = {
                "start": start,
                "end": end,
                "peptide": peptide_windows[i],
                "probability": probabilities[i],
                "label": labels[i],
            }
            
            if args.properties:
                result.update(properties[i])
            
            results.append(result)
        
        # Calculate overlapping regions
        hotspots.sort(key=lambda x: x["start"])
        for i in range(len(hotspots)):
            for j in range(i+1, len(hotspots)):
                # Check overlap: current hotspot end > next hotspot start
                if hotspots[i]["end"] > hotspots[j]["start"]:
                    hotspots[i]["overlap_count"] += 1
                    hotspots[j]["overlap_count"] += 1
        
        # Display results
        print("\n📊 Scan Results:")
        print("=" * 80)
        print(f"🔹 Window size: {window_size} aa")
        print(f"🔹 Hotspot threshold: {args.threshold}")
        print(f"🔴 Hotspot regions identified: {hotspot_count}")
        
        if hotspots:
            print("\n🔥 Hotspot regions overview:")
            # Sort by overlap count (most overlapping first)
            hotspots.sort(key=lambda x: (-x["overlap_count"], x["start"]))
            
            for i, hs in enumerate(hotspots[:args.limit]):
                print(f"{i+1}. Position: {hs['start']}-{hs['end']}")
                print(f"   Peptide: {hs['peptide']}")
                print(f"   Predicted probability: {hs['probability']:.4f}")
                print(f"   Overlap count: {hs['overlap_count']}")
                print("-" * 50)
        
        # Save results
        if args.output:
            save_results_to_csv(results, args.output)
                
    except Exception as e:
        print(f"⚠️ Scan failed: {str(e)}")

# ====================== Main Program ========================
def main():
    """Main function, parse command-line arguments and execute appropriate functionality"""
    # Create main parser
    parser = argparse.ArgumentParser(
        description="🧬 MS Inducing Peptide Prediction System - CLI Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        help="select operation to perform"
    )

    # ===== Predict Command =====
    predict_parser = subparsers.add_parser(
        "predict",
        help="predict MS induction risk for peptides"
    )
    predict_group = predict_parser.add_mutually_exclusive_group(required=True)
    predict_group.add_argument(
        "-s", "--sequence",
        help="input a single peptide sequence"
    )
    predict_group.add_argument(
        "-f", "--file",
        help="input file containing multiple sequences (one per line)"
    )
    predict_group.add_argument(
        "--fasta",
        help="input FASTA file"
    )
    predict_parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.7,
        help="set probability threshold (0.5-1.0)"
    )
    predict_parser.add_argument(
        "-p", "--properties",
        action="store_true",
        help="display physicochemical properties"
    )
    predict_parser.add_argument(
        "-o", "--output",
        help="save results to CSV file"
    )
    predict_parser.add_argument(
        "-l", "--limit",
        type=int,
        default=5,
        help="limit number of results displayed on screen"
    )

    # ===== Design Command =====
    design_parser = subparsers.add_parser(
        "design",
        help="design and analyze single-point mutants"
    )
    design_parser.add_argument(
        "-s", "--sequence",
        required=True,
        help="input peptide sequence"
    )
    design_parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.7,
        help="set probability threshold (0.5-1.0)"
    )
    design_parser.add_argument(
        "-p", "--properties",
        action="store_true",
        help="display physicochemical properties"
    )
    design_parser.add_argument(
        "-o", "--output",
        help="save results to CSV file"
    )
    design_parser.add_argument(
        "-l", "--limit",
        type=int,
        default=10,
        help="limit number of high-risk mutants displayed"
    )

    # ===== Scan Command =====
    scan_parser = subparsers.add_parser(
        "scan",
        help="scan protein sequences for hotspot regions"
    )
    scan_parser.add_argument(
        "--fasta",
        required=True,
        help="input protein FASTA file"
    )
    scan_parser.add_argument(
        "-w", "--window",
        type=int,
        default=9,
        help="set sliding window size (4-50)"
    )
    scan_parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.7,
        help="set hotspot probability threshold (0.5-1.0)"
    )
    scan_parser.add_argument(
        "-p", "--properties",
        action="store_true",
        help="display physicochemical properties"
    )
    scan_parser.add_argument(
        "-o", "--output",
        help="save results to CSV file"
    )
    scan_parser.add_argument(
        "-l", "--limit",
        type=int,
        default=5,
        help="limit number of hotspots displayed"
    )

    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if args.threshold and (args.threshold < 0.5 or args.threshold > 1.0):
        print("⚠️ Warning: Threshold should be between 0.5-1.0, using default value 0.7")
        args.threshold = 0.7
    
    # Execute appropriate command
    if not hasattr(args, 'command'):
        parser.print_help()
        return
    
    command_handlers = {
        "predict": run_predict_mode,
        "design": run_design_mode,
        "scan": run_scan_mode
    }
    
    handler = command_handlers.get(args.command)
    if handler:
        handler(args)
    else:
        print(f"⚠️ Error: Unknown command '{args.command}'")

if __name__ == "__main__":
    main()