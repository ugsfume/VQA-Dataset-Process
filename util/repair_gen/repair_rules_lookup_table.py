import json

class RepairRules:
    def __init__(self, file_dir: str = "repair_rules.json"):

        self.repair_rules = json.load(open(file_dir, 'r', encoding='utf-8'))
        self._table = {
            ('TOPEN', 'Gate'): [self.repair_rules["TOPEN-Gate"]],
            ('TOPEN', 'Data'): [self.repair_rules["TOPEN-Data"]],
            ('TOPEN', 'Gate&Data'): [self.repair_rules["TOPEN-Gate"],self.repair_rules["TOPEN-Data"]],
            ('TOPEN', 'Gate&Source'): [self.repair_rules["暗点"]],
            ('TOPEN', 'Data&Data'): [self.repair_rules["不修补"]],
            ('TOPEN', 'Data&Com'): [self.repair_rules["TOPEN-Data&Com"]],
            ('TOPEN', 'Gate&Gate'): [self.repair_rules["不修补"]],
            ('TOPEN', 'Gate&Com'): [self.repair_rules["不修补"]],
            ('TOPEN', 'Gate&TFT'): [self.repair_rules["TOPEN-Gate"],self.repair_rules["暗点"]],
            ('TOPEN', 'Gate&Drain'): [self.repair_rules["TOPEN-Gate"],self.repair_rules["暗点"]],
            ('TOPEN', 'Gate&Mesh'): [self.repair_rules["不修补"]],
            ('TOPEN', 'TFT'): [self.repair_rules["暗点"]],
            ('TOPEN', 'Data&Drain'): [self.repair_rules["不修补"]],
            ('TOPEN', 'Data&Mesh'): [self.repair_rules["不修补"]],
            ('TOPEN', 'Data&ITO'): [self.repair_rules["不修补"]],
            ('TOPEN', 'Gate&ITO'): [self.repair_rules["不修补"]],
            ('TOPEN', 'Drain'): [self.repair_rules["暗点"]],
            ('TOPEN', 'Source'): [self.repair_rules["暗点"]],
            ('TOPEN', 'ITO'): [self.repair_rules["TOPEN-ITO"]],
            ('TOPEN', 'Mesh_Hole'): [self.repair_rules["不修补"]],
            ('TOPEN', 'VIA_Hole'): [self.repair_rules["暗点"]],
            ('TOPEN', 'Com'): [self.repair_rules["不修补"]],

            ('TSHRT', 'Gate'): [self.repair_rules["TSHRT-Gate"]],
            ('TSHRT', 'Data'): [self.repair_rules["TOPEN-Data"]],
            ('TSHRT', 'Gate&Data'): [self.repair_rules["TSHRT-Gate&Data"]],
            ('TSHRT', 'Gate&Source'): [self.repair_rules["暗点"]],
            ('TSHRT', 'Data&Data'): [self.repair_rules["TSHRT-Data&Data"]],
            ('TSHRT', 'Data&Com'): [self.repair_rules["TSHRT-Data&Com"]],
            ('TSHRT', 'Gate&Gate'): [self.repair_rules["TSHRT-Gate&Gate"]],
            ('TSHRT', 'Gate&Com'): [self.repair_rules["TSHRT-Gate&Com"]],
            ('TSHRT', 'Gate&TFT'): [self.repair_rules["TOPEN-Gate"],self.repair_rules["暗点"]],
            ('TSHRT', 'Gate&Drain'): [self.repair_rules["TOPEN-Gate"],self.repair_rules["暗点"]],
            ('TSHRT', 'Gate&Mesh'): [self.repair_rules["Mesh"]],
            ('TSHRT', 'TFT'): [self.repair_rules["暗点"]],
            ('TSHRT', 'Data&Drain'): [self.repair_rules["暗点"]],
            ('TSHRT', 'Data&Mesh'): [self.repair_rules["Mesh"]],
            ('TSHRT', 'Data&ITO'): [self.repair_rules["TSHRT-Data&ITO"]],
            ('TSHRT', 'Gate&ITO'): [self.repair_rules["TSHRT-Gate&ITO"]],
            ('TSHRT', 'Drain'): [self.repair_rules["暗点"]],
            ('TSHRT', 'Source'): [self.repair_rules["暗点"]],
            ('TSHRT', 'ITO'): [self.repair_rules["TSHRT-ITO"]],
            ('TSHRT', 'Mesh_Hole'): [self.repair_rules["Mesh"]],
            ('TSHRT', 'VIA_Hole'): [self.repair_rules["暗点"]],
            ('TSHRT', 'Com'): [self.repair_rules["不修补"]],

            ('TSMRN', 'Gate'): [self.repair_rules["不修补"]],
            ('TSMRN', 'Data'): [self.repair_rules["不修补"]],
            ('TSMRN', 'Gate&Data'): [self.repair_rules["不修补"]],
            ('TSMRN', 'Gate&Source'): [self.repair_rules["暗点"]],
            ('TSMRN', 'Data&Data'): [self.repair_rules["TSHRT-Data&Data"]],
            ('TSMRN', 'Data&Com'): [self.repair_rules["不修补"]],
            ('TSMRN', 'Gate&Gate'): [self.repair_rules["TSHRT-Gate&Gate"]],
            ('TSMRN', 'Gate&Com'): [self.repair_rules["TSHRT-Gate&Com"]],
            ('TSMRN', 'Gate&TFT'): [self.repair_rules["不修补"]],
            ('TSMRN', 'Gate&Drain'): [self.repair_rules["暗点"]], # 
            ('TSMRN', 'Gate&Mesh'): [self.repair_rules["不修补"]],
            ('TSMRN', 'TFT'): [self.repair_rules["暗点"]],
            ('TSMRN', 'Data&Drain'): [self.repair_rules["暗点"]],
            ('TSMRN', 'Data&Mesh'): [self.repair_rules["Mesh"]],
            ('TSMRN', 'Data&ITO'): [self.repair_rules["TSMRN-Data&ITO"]],
            ('TSMRN', 'Gate&ITO'): [self.repair_rules["TSMRN-Gate&ITO"]],
            ('TSMRN', 'Drain'): [self.repair_rules["暗点"]],
            ('TSMRN', 'Source'): [self.repair_rules["暗点"]],
            ('TSMRN', 'ITO'): [self.repair_rules["TSMRN-ITO"]],
            ('TSMRN', 'Mesh_Hole'): [self.repair_rules["Mesh"]],
            ('TSMRN', 'VIA_Hole'): [self.repair_rules["暗点"]],
            ('TSMRN', 'Com'): [self.repair_rules["不修补"]],
        }

    def get_value(self, key1, key2):
        if key1 in ['TSXSP']:
            return [self.repair_rules["不修补"]]
        if key1 in ['TSCOK']:
            return [self.repair_rules["TSCOK"]]
        if key1 in ['TSCNG']:
            return [self.repair_rules["TSCNG"]]
        
        """Retrieve a value using two keys."""
        return self._table.get((key1, key2), None)  # Returns None if key doesn't exist
    

# lookup = RepairRules()
# rule = lookup.get_value('TSMRN', 'Gate&Data')
# print(rule)