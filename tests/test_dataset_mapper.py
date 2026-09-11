from pathlib import Path
import pytest

from src.dataset.mapper import ClassTaxonomyMapper

ESC50_CLASSES = {
    'airplane', 'breathing', 'brushing_teeth', 'can_opening', 'car_horn', 'cat', 'chainsaw',
    'chirping_birds', 'church_bells', 'clapping', 'clock_alarm', 'clock_tick', 'coughing', 'cow',
    'crackling_fire', 'crickets', 'crow', 'crying_baby', 'dog', 'door_wood_creaks', 'door_wood_knock',
    'drinking_sipping', 'engine', 'fireworks', 'footsteps', 'frog', 'glass_breaking', 'hand_saw',
    'helicopter', 'hen', 'insects', 'keyboard_typing', 'laughing', 'mouse_click', 'pig',
    'pouring_water', 'rain', 'rooster', 'sea_waves', 'sheep', 'siren', 'sneezing', 'snoring',
    'thunderstorm', 'toilet_flush', 'train', 'vacuum_cleaner', 'washing_machine', 'water_drops', 'wind'
}


@pytest.fixture
def mapper():
    cfg_path = Path(__file__).resolve().parent.parent / 'configs' / 'class_mapping.yaml'
    return ClassTaxonomyMapper(cfg_path)


def test_mapper_initialization(mapper):
    assert len(mapper.class_to_target) > 0
    assert len(mapper.target_id_to_name) == 3
    assert mapper.target_id_to_name[0] == 'stationary'
    assert mapper.target_id_to_name[1] == 'non_stationary'
    assert mapper.target_id_to_name[2] == 'impulsive'


def test_specific_class_mappings(mapper):
    assert mapper.map_category('wind') == 0
    assert mapper.map_category('vacuum_cleaner') == 0
    assert mapper.map_category('rain') == 0

    assert mapper.map_category('siren') == 1
    assert mapper.map_category('crying_baby') == 1
    assert mapper.map_category('chirping_birds') == 1

    assert mapper.map_category('door_wood_knock') == 2
    assert mapper.map_category('glass_breaking') == 2
    assert mapper.map_category('mouse_click') == 2


def test_excluded_classes(mapper):
    assert mapper.map_category('crackling_fire') is None
    assert mapper.map_category('toilet_flush') is None
    assert mapper.map_category('cat') is None
    assert mapper.map_category('non_existent_xyz') is None


def test_no_orphaned_classes(mapper):
    mapped = set(mapper.class_to_target.keys())
    excluded = mapper.excluded_classes
    accounted = mapped | excluded
    orphaned = ESC50_CLASSES - accounted
    assert len(orphaned) == 0, f"Orphaned classes: {orphaned}"


def test_no_invalid_class_names(mapper):
    mapped = set(mapper.class_to_target.keys())
    excluded = mapper.excluded_classes
    all_referenced = mapped | excluded
    invalid = all_referenced - ESC50_CLASSES
    assert len(invalid) == 0, f"Invalid class names: {invalid}"


def test_class_balance_summary(mapper):
    summary = mapper.get_summary()
    assert summary['stationary'] >= 8
    assert summary['non_stationary'] >= 8
    assert summary['impulsive'] >= 8
