import pytest
import numpy as np
import sys
sys.path.insert(0, r'\\wsl$\Ubuntu\home\shaheera\mnist-fc')


def data_type_normalize(datatype, NormalizationValue):
    input_image = np.random.rand(1, 784)
    expected_type_image = input_image.astype(datatype)
    expected_type_image_normalize = expected_type_image/NormalizationValue
    result = input_image.astype(datatype)
    result_normalize = result/NormalizationValue
    if datatype == 'int32' or NormalizationValue != 255 or datatype == 'str':
        raise Exception("Invalid datatype and Normalization Value")
    return expected_type_image, expected_type_image_normalize, result, result_normalize

#Component 1 - positive case
def test_data_type_normalize_positive():
        expected_type_image, expected_type_image_normalize, result, result_normalize = data_type_normalize('float32', 255)      
        assert np.array_equal(expected_type_image, result)
        assert np.array_equal(expected_type_image_normalize, result_normalize)
        
#Component 2 - Exception case
@pytest.mark.parametrize("value", [150, 300, 1])
def test_data_type_normalize_exception(value):
    with pytest.raises(Exception) as exc_info:
        expected_type_image, expected_type_image_normalize, result, result_normalize = data_type_normalize('int32', value)
    assert exc_info.type is Exception
    assert exc_info.value.args[0] == "Invalid datatype and Normalization Value"

    

    
