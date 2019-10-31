package ng

import (
	"bytes"
	"fmt"
	"github.com/sudachen/go-dnn/mx"
	"github.com/sudachen/go-dnn/nn"
	"gopkg.in/yaml.v3"
	"io/ioutil"
	"os"
	"path/filepath"
)

const StateYml = "state.yml"
const ParamsFmt = "params-%d.dat"
const NetworkYml = "network.yml"
const SummaryTxt = "summary.txt"

type EpochInfo struct {
	Accuracy float32   `yaml:"accuracy"`
	Detail   []float32 `yaml:"detail"`
	AvgLoss  float32   `yaml:"aloss"`
	LastLoss float32   `yaml:"lloss"`
}

type GymInfo struct {
	Identity  string            `yaml:"identity"`
	Seed      int               `yaml:"seed"`
	Epoch     int               `yaml:"last"`
	Summary   mx.Summary	    `yaml:"-"`
	EpochInfo map[int]EpochInfo `yaml:"epochs"`
}

func (g *GymInfo) Exists(dir string) bool {
	fname := filepath.Join(dir, StateYml)
	if st, err := os.Stat(fname); err == nil && ((st.Mode() & os.ModeType) == 0) {
		return true
	}
	return false
}

func (g *GymInfo) Init(net *nn.Network, seed int) {
	g.Epoch = 0
	g.EpochInfo = map[int]EpochInfo{}
	g.Identity = net.Identity().String()
	g.Seed = seed
}

func (g *GymInfo) Update(epoch int, acc float32, detail []float32, al AvgLoss) {
	ei := EpochInfo{
		Accuracy: acc,
		Detail:   detail,
		AvgLoss:  al.Value(),
		LastLoss: al.Last(),
	}
	g.EpochInfo[epoch] = ei
	g.Epoch = epoch
}

func (g *GymInfo) Load(dir string) (err error) {
	var bs []byte
	fname := filepath.Join(dir, StateYml)
	if bs, err = ioutil.ReadFile(fname); err != nil {
		return
	}
	if err = yaml.Unmarshal(bs, g); err != nil {
		return
	}
	return nil
}

func (g *GymInfo) Save(dir string) (err error) {
	var bs []byte
	fname := filepath.Join(dir, StateYml)
	if err = os.MkdirAll(dir, 0777); err != nil {
		return
	}
	if bs, err = yaml.Marshal(g); err != nil {
		return
	}
	if err = ioutil.WriteFile(fname, bs, 0666); err != nil {
		return
	}
	return nil
}

func (gs *GymInfo) SaveParams(epoch int, net *nn.Network, dir string) (err error) {
	fname := filepath.Join(dir, fmt.Sprintf(ParamsFmt, epoch))
	if err = os.MkdirAll(dir, 0777); err != nil {
		return
	}
	return net.SaveParamsFile(fname)
}

func (gs *GymInfo) LoadParams(epoch int, net *nn.Network, dir string) (err error) {
	fname := filepath.Join(dir, fmt.Sprintf(ParamsFmt, epoch))
	return net.LoadParamsFile(fname, false)
}

func (g *GymInfo) WriteNetwork(net *nn.Network, dir string) (err error) {
	var bs []byte
	var summary mx.Summary
	fname := filepath.Join(dir, NetworkYml)
	if err = os.MkdirAll(dir, 0777); err != nil {
		return
	}
	if summary, err = net.Summary(false); err != nil {
		return
	}
	if bs, err = yaml.Marshal(&summary); err != nil {
		return
	}
	if err = ioutil.WriteFile(fname, bs, 0666); err != nil {
		return
	}
	bf := bytes.Buffer{}
	summary.Print(func(s string){ bf.WriteString(s+"\n")})
	fname = filepath.Join(dir, SummaryTxt)
	if err = ioutil.WriteFile(fname, bf.Bytes(), 0666); err != nil {
		return
	}
	return nil
}
