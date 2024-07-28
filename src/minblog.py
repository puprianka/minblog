# Import packages
import argparse as ap
import numpy as np
from torch_geometric.data import Data
from pathlib import Path
import torch
from graph import make_graph
from utils import load_pla, check_implies, write_pla, check_cover
import time
import csv

torch_device = torch.device("cpu")

def main():

    parser = ap.ArgumentParser()
    parser.add_argument("-indir", required=False, default=None, help="Provide dir path with PLA files")
    parser.add_argument("-outdir", required=False, default=None, help="Provide output dir path for saving minPLA files")
    parser.add_argument("-model", required=False, default="modelv4", help="Provide dir path for model")
    args = parser.parse_args()

    # Load Model
    model_file = Path("gat_model.pt")
    weight_file = Path("gat_weight.pt")

    # Load model if available
    if model_file.is_file() and weight_file.is_file():
        print(f"Loading trained model from {model_file}")
        gat_model = torch.load(model_file)
        gat_model.load_state_dict(torch.load(weight_file))
        gat_model.eval()
        gat_model.to(torch_device)
    else:
        print(f"Model files dont exist. Cant proceed")
        exit()

    # Load PLA files from PLA input dir
    out_db = []
    for pla_file in Path(args.indir).glob('*.pla'):
        stats = {'testcase': pla_file.stem}
        # Check if its already a minfile
        if 'min' in pla_file.stem:
            continue
        print(f"Reading {pla_file.stem}")
        read_start = time.time()
        func_matrix = load_pla(pla_file=pla_file)
        #TODO: handle fr fd f type
        ifunc_matrix = func_matrix[func_matrix[:, -1] == 0, :]
        read_end = time.time()
        # Save file read time
        stats['read_time'] = read_end-read_start

        # Make Graph
        print(f"Building Graph")
        graph_start = time.time()
        sop_graph, node_dict = make_graph(func_matrix, ifunc_matrix)

        # Convert to pygdata
        x = torch.tensor([node['feature'] for node in sop_graph.nodes()], dtype=torch.long).float()
        edge_list = np.array(sop_graph.edge_list())
        edge_index = np.vstack([edge_list[:, 0], edge_list[:, 1]], dtype=np.int32)
        pyg_graph = Data(x=x, edge_index=torch.tensor(edge_index))
        graph_end = time.time()
        stats['graph_time'] = graph_end-graph_start

        # call model for minpred
        print(f"Minimizing")
        pred_start = time.time()
        output = gat_model(pyg_graph.x, pyg_graph.edge_index)
        pred_end = time.time()
        stats['pred_time'] = pred_end-pred_start

        # Validate and correct
        imply_start = time.time()
        class_1_idx = torch.argmax(output, dim=1).tolist()
        correct_pred_imp = []
        missing_imp = []

        # Evaluate every prediction by doing pf' check
        print(f"Validating Solution")
        for node_idx, cls1 in enumerate(class_1_idx):
            if cls1 == 1:
                ntype = sop_graph[node_idx]['type']
                nexpr = sop_graph[node_idx]['imp']                
                # Check which one is wrong using pf' check:
                if check_implies(ifunc_matrix[:, :-1], nexpr): 
                    correct_pred_imp.append(nexpr.tolist() + [1])
                if check_cover(func_matrix[:, :-1], nexpr):
                    missing_imp.append(nexpr)

        imply_stop = time.time()
        stats['imply_time'] = imply_stop-imply_start
        
        # Save the predictions as a pla file
        print(f"Writing output")
        save_start = time.time()
        out_pla = Path(args.outdir) / f'{pla_file.stem}_minblog.pla'
        ofunc_matrix = np.array(correct_pred_imp)
        write_pla('fr', ofunc_matrix, out_pla)
        save_end = time.time()
        stats['save_time'] = save_end-save_start
        stats['total'] = save_end-read_start

        #TODO: Save data about post lit and pre lit
        stats['pre_cubes'] = func_matrix.shape[0]
        stats['pre_lits'] = np.count_nonzero(func_matrix[:, :-1] < 2)
        stats['post_cubes'] = ofunc_matrix.shape[0]
        stats['post_lits'] = np.count_nonzero(ofunc_matrix[:, :-1] < 2)

        out_db.append(stats.copy())
        print(f"Total time: {stats['total']:0.2f} Seconds")
    
    # Save outdb as csv
    with open('minblog_runtime.csv', 'w') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_db[0].keys()))
        # Write header
        writer.writeheader()
        writer.writerows(out_db)


if __name__ == '__main__':
    main()
