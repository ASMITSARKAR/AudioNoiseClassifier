from pathlib import Path
import yaml


class ClassTaxonomyMapper:
    def __init__(self, mapping_config_path):
        self.config_path = Path(mapping_config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.raw_config = yaml.safe_load(f)

        self.class_to_target = {}
        self.class_to_target_name = {}
        self.target_id_to_name = {}
        self.target_name_to_id = {}
        self.excluded_classes = set(self.raw_config.get('excluded_classes', []))
        self.class_justifications = {}
        self._build_mappings()

    def _build_mappings(self):
        targets = ['stationary', 'non_stationary', 'impulsive']
        for name in targets:
            if name not in self.raw_config:
                continue
            section = self.raw_config[name]
            tid = int(section['target_id'])
            self.target_id_to_name[tid] = name
            self.target_name_to_id[name] = tid

            for item in section.get('classes', []):
                cname = item['name']
                self.class_to_target[cname] = tid
                self.class_to_target_name[cname] = name
                self.class_justifications[cname] = item.get('justification', '')

    def map_category(self, category):
        return self.class_to_target.get(category, None)

    def get_target_name(self, target_id):
        return self.target_id_to_name.get(target_id, 'unknown')

    def get_summary(self):
        summary = {name: 0 for name in self.target_name_to_id}
        for _, t_name in self.class_to_target_name.items():
            if t_name in summary:
                summary[t_name] += 1
        return summary
