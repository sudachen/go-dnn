package ng

import "fmt"

func Classification(data, label []float32) (bool, error) {
	if len(label) != 1 {
		return false, fmt.Errorf("Classify must have label parameter as slice with one value")
	}
	index := int(label[0])
	if len(data) < index || index < 0 {
		return false, fmt.Errorf("Classify label index out of data range")
	}
	maxindex := index
	for i, v := range data {
		if v > data[maxindex] {
			maxindex = i
		}
	}
	return maxindex == index, nil
}
