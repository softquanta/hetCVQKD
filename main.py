# main.py
# Copyright 2020 Alexandros Georgios Mountogiannakis

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import heterodyne
import generic
import numpy as np
import os


# Create necessary folders if they do not exist
if not os.path.exists('logs'):
    os.makedirs('logs')
if not os.path.exists('keys'):
    os.makedirs('keys')
if not os.path.exists('codes'):
    os.makedirs('codes')
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('betas'):
    os.makedirs('betas')

config = "normal"  # Available options: "normal", find_ideal_beta", "optimize_beta", "optimize_pe"

vals = heterodyne.Protocol(n_bks=5, N=5000, M=0.1, mu=30, A=0.2, L=3, T=None, eta=0.85, v_el=0.1, xi=0.01,
                           beta=0.65, iter_max=150, p=8, q=4, alpha=7, p_EC_tilde=1, protocol='heterodyne',
                           ec=True, pe_alt=True, load_code=False, save_data=False, load_data=False, save_pe=False,
                           load_pe=False, optimal_mu=False)

# The normal mode for a simulation.
if config == "normal":
    heterodyne.Protocol.validity_checks(vals)
    results = heterodyne.Protocol.processing(vals)
    heterodyne.Protocol.file_logging(vals, results)

elif config == "find_ideal_beta":
    # Input
    vals.b_id = True
    load_beta = False
    runs = 100
    desired_code_rate = 1 - 2 / 12
    betas_file = "betas/N_" + str(vals.N) + "_n_bks_" + str(vals.n_bks) + "_p_" + str(vals.p) + "L1.npz"
    print("Desired rate:", desired_code_rate)

    if not load_beta:
        betas = np.zeros(shape=runs, dtype=np.float64)
        for j in range(runs):
            print("Run", j + 1, "starting...")
            heterodyne.Protocol.validity_checks(vals)
            r, i, h, q, p, d_ent, prot = heterodyne.Protocol.processing(vals)
            b = generic.precise_reconciliation_efficiency(desired_code_rate, i, h, q, p, d_ent)
            if b > 1:
                raise RuntimeError("The specified desired code rate returns a reconciliation efficiency above 1.")
            else:
                betas[j] = b
            print("Run", j + 1, "Beta:", b, "Average beta until now:", np.average(betas[np.argwhere(betas)]), "\n")
        print("The reconciliation efficiency to use given these certain parameters is:", np.average(betas))
        np.savez_compressed(betas_file, beta=betas)
    else:
        data = np.load(betas_file, allow_pickle=True)
        betas = data["beta"]
        print(np.average(betas))

elif config == "optimize_beta":
    vals.b_id = True
    wc = 2
    rates = list()
    betas = dict()

    heterodyne.Protocol.validity_checks(vals)
    r, i, h, q, p, d_ent, prot = heterodyne.Protocol.processing(vals)
    for j in range(50):
        wr = wc + 1 + j  # For a positive code rate, the minimum possible row weight is the column weight plus one
        rate = 1 - wc / wr
        rates.append(rate)
        b = generic.precise_reconciliation_efficiency(rates[j], i, h, q, p, d_ent)
        if b >= 1:  # Once the value of beta goes above one, the maximum possible row weight is found and the loop stops
            break
        elif b < 0:  # Negative betas are not added to the list of possible choices
            continue
        else:
            betas[wr] = (b, rates[j])
    for k, d in betas.items():
        print(str(k) + ":", str(d))

elif config == "optimize_pe":
    m = np.arange(0.01, 1, 0.01)  # Examine every M from 1% to 99% with step 1%
    m = np.around(m, decimals=2)  # Eliminate numerical inaccuracies
    r = np.zeros_like(m, dtype=np.float64)  # Storage for composable key rates
    for i in range(len(m)):
        print("PE Percentage:", int(m[i] * 100), "%")
        vals = heterodyne.Protocol(n_bks=50, N=50000, M=m[i], mu=30, A=0.2, L=3, T=None, eta=0.85, v_el=0.1, xi=0.01,
                                   beta=0.65, iter_max=150, p=8, q=4, alpha=7, p_EC_tilde=1, protocol='heterodyne',
                                   ec=False, pe_alt=True, load_code=False, save_data=False, load_data=False,
                                   save_pe=False, load_pe=False, optimal_mu=False)
        heterodyne.Protocol.validity_checks(vals)
        results = heterodyne.Protocol.processing(vals)
        r[i] = results.R_theo
    print(max(r))

else:
    raise RuntimeError("This configuration does not exist. Please choose from an existing configuration.")
