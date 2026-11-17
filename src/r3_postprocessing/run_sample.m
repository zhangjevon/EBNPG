% Added for the EBNPG manuscript (third revision).
% SPDX-License-Identifier: GPL-3.0-or-later
% Independent re-certification of a systematic sample (every 25th point).
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));   % <root>
pl = {'AP1','AP3','AP4','BK1','DD1','DGO2','DTLZ2','FA1','Far1','FDS','Hil1','IKK1','KW2','Lov4','Lov5','Lov6','LTDZ','MGH16','MGH26','MGH33','MLF2','MMR1','MMR2','MMR3','MOP5','MOP6','MOP7','NIC1','NIC2','NIC3','NIM1','NSP1','NSP2','PNR','QV1','SK2','SP1','Toi4','Toi9','VU2','ZDT6','ZLT1'};
cert_sample(pl, 25, fullfile(root,'Code_Data','cert_sample.csv'));
