package feature

import (
	"math"
	"sort"
	"unsafe"
)

const SeriesReserved = 64*1024/unsafe.Sizeof(SeriesItem{}) // 64K

// 24 bytes
type SeriesItem struct {
	Index  int64
	Value  float32
	Value2 float32
	Count  int32
	_      int32
}

type Series struct {
	Items []SeriesItem
	First,Last  int64
}

func (ser *Series) Update(index int64, value float32) {
	ser.UpdateEx(index,value, 0, nil)
}

func (ser *Series) UpdateMin(index int64, value float32) {
	ser.UpdateEx(index,value, 0, func(v *SeriesItem){
		if v.Value > value {
			v.Value = value
		}
	})
}

func (ser *Series) UpdateMax(index int64, value float32) {
	ser.UpdateEx(index,value, 0, func(v *SeriesItem){
		if v.Value < value {
			v.Value = value
		}
	})
}

func (ser *Series) UpdateAvg(index int64, value float32) {
	ser.UpdateEx(index,value, 0, func(v *SeriesItem){
		v.Value = float32((float64(v.Value)*float64(v.Count)+float64(value))/float64(v.Count+1))
		v.Count++
	})
}

func (ser *Series) UpdateSum(index int64, value float32) {
	ser.UpdateEx(index,value, 0, func(v *SeriesItem){
		v.Value += value
		v.Count++
	})
}

func (ser *Series) At(index int64) (SeriesItem, bool) {
	L := len(ser.Items)
	if L != 0 && ser.First == index {
		return ser.Items[0], true
	}
	if L != 0 && ser.Last == index {
		return ser.Items[L-1], true
	}
	if L != 0 {
		i := sort.Search(L,func(n int)bool{
			return ser.Items[n].Index >= index
		})
		if i < L && i > 0 && ser.Items[i].Index == index {
			return ser.Items[i], true
		}
	}
	return SeriesItem{}, false
}

func (ser *Series) UpdateEx(index int64, value, v2 float32, update func(*SeriesItem)) {
	L := len(ser.Items)
	if L == 0 {
		ser.Allocate()
		ser.Items = append(ser.Items,SeriesItem{index,value, v2, 1, 0})
		ser.First = index
		ser.Last = index
	} else if /*ser.Len > 0 &&*/ index < ser.First {
		ser.Items = append(ser.Items,SeriesItem{})
		copy(ser.Items[1:],ser.Items[:L])
		ser.Items[0] = SeriesItem{index,value, v2, 1, 0}
		ser.First = index
	} else if /*ser.Len > 0 &&*/ index > ser.Last {
		ser.Items = append(ser.Items, SeriesItem{index, value, v2, 1, 0})
		ser.Last = index
	} else if index == ser.Last {
		if update != nil {
			update(&ser.Items[L-1])
		} else {
			ser.Items[L-1] = SeriesItem{index,value, v2, 1, 1}
		}
	} else {
		i := sort.Search(L,func(n int)bool{
			return ser.Items[n].Index >= index
		})
		if i < L && ser.Items[i].Index == index {
			if update != nil {
				update(&ser.Items[i])
			} else {
				ser.Items[i] = SeriesItem{index, value, v2, 1, 1}
			}
		} else {
			ser.Items = append(ser.Items,SeriesItem{})
			copy(ser.Items[i+1:],ser.Items[i:])
			ser.Items[i] = SeriesItem{index,value, v2, 1, 0}
		}
	}
}

func (ser *Series) Allocate() {
	ser.Items = make([]SeriesItem,0,SeriesReserved)
}

func (ser *Series) Len() int {
	return len(ser.Items)
}

func (ser *Series) Resample(
	indexer func(int64)int64,
	sampler func(int64,[]SeriesItem),
	window int) {

	var pos []int
	var index []int64

	if len(ser.Items) > 0 {
		nd := indexer(ser.First)
		pos = append(pos,0)
		index = append(index, nd)

		for i,v := range ser.Items {
			if x := indexer(v.Index); x != nd {
				nd = x
				pos = append(pos,i)
				index = append(index,nd)
			}
		}

		for i,dx := range index {
			k := window
			j := pos[i]
			e := len(ser.Items)
			if i+1 < len(pos) {
				e = pos[i+1]
			}
			if i > k-1 {
				j = pos[i-k+1]
			}
			sampler(dx,ser.Items[j:e])
		}
	}
}

func Merge(merge func(index int64, item []SeriesItem, ok []bool), a ...Series) {
	i := make([]int,len(a))
	item := make([]SeriesItem,len(a))
	ok := make([]bool,len(a))

	for {
		k := int64(math.MaxInt64)
		for j,v := range a {
			ii := i[j]
			if ii < v.Len() && v.Items[ii].Index < k {
				k = v.Items[ii].Index
			}
		}
		if k == math.MaxInt64 {
			break
		}
		for j,v := range a {
			ii := i[j]
			if ii < v.Len() && v.Items[ii].Index == k {
				item[j] = v.Items[ii]
				ok[j] = true
				i[j]++
			} else {
				item[j] = SeriesItem{}
				ok[j] = false
			}
		}
		merge(k, item, ok)
	}
}
