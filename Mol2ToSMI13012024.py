from rdkit import Chem
from rdkit.Chem import AllChem
import os
from openbabel import openbabel
from openbabel import pybel
from openbabel.pybel import (readfile,Outputfile) 
import numpy as np
import csv
'''
def MolFormatConversion(input_file:str,output_file:str,input_format="mol2",output_format="smi"):
    molecules = readfile(input_format,input_file)
    output_file_writer = Outputfile(output_format,output_file)
    for i,molecule in enumerate(molecules):
        output_file_writer.write(molecule)
        output_file_writer.close()
        print('%d molecules converted'%(i+1))

def convert_mol2_to_smi(mol2_file, smi_file):
    # 创建OBConversion对象
    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats("mol2", "smi")

    # 创建OBMol对象
    mol = openbabel.OBMol()

    # 打开.mol2文件
    if conv.ReadFile(mol, mol2_file):
        # 创建.smi文件并打开以写入SMILES
        with open(smi_file, 'w') as smi_file:
            # 遍历分子并写入SMILES
            for idx, atom in enumerate(mol.GetAtom()):
                smi = conv.WriteString(mol, idx, False)
                smi_file.write(f"{smi.strip()},1.1\n")

def convert_mol2_to_smiles(mol2_file):
    # 创建OBConversion对象
    conv = openbabel.OBConversion()
    conv.SetInFormat("mol2")

    # 创建OBMol对象
    mol = openbabel.OBMol()
    
    # 打开.mol2文件
    if conv.ReadFile(mol, mol2_file):
        # 生成SMILES字符串
        smiles = conv.WriteString(mol)
        return smiles.strip()
    else:
        print(f"无法读取文件 {mol2_file}")
        return 'NONE'
'''


smi_file_path = os.path.join('datasets', 'predict.smi')
smi_id_path = os.path.join('datasets', 'predict_id.smi')
smiles_result = []
smiles_withid = []#14250
for i in range(1,100):
    molid = "MOL"+"{:06d}".format(i)
    molpath = molid+".mol2"
    molfile = os.path.join('mol2_files',molpath)
    if not os.path.exists(molfile):
        continue
    mol2 = readfile("mol2",molfile)

    for idx,mol in enumerate(mol2):
        replaced = '	'
        smiles = mol.write("smi")
        smiles_withid.append(smiles)
        smiles_replaced = smiles.replace(replaced,',')
        smiles_result.append(smiles_replaced)

with open(smi_file_path, 'w') as smi_file:
    # 遍历分子并写入SMILES
    for idx,smi in enumerate(smiles_result):
        smi_file.write(f"{smi.strip()},1.1\n")
with open(smi_id_path, 'w') as smi_file:
    # 遍历分子并写入SMILES
    for idx,smi in enumerate(smiles_withid):
        smi_file.write(f"{smi.strip()},1.1\n")
