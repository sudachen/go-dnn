package nn

import (
	"github.com/sudachen/go-dnn/mx"
)

type Block interface {
	Combine(*mx.Symbol, ...*mx.Symbol) (*mx.Symbol, []*mx.Symbol, error)
}

type BlockConnect struct {
	blocks []Block
}

func (bc *BlockConnect) Combine(s *mx.Symbol, g ...*mx.Symbol) (*mx.Symbol, []*mx.Symbol, error) {
	var err error
	for _, b := range bc.blocks {
		if s, g, err = b.Combine(s, g...); err != nil {
			return nil, nil, err
		}
	}
	return s, g, nil
}

func Connect(b ...Block) Block {
	return &BlockConnect{b}
}

type BlockConcat struct {
	blocks []Block
}

func (bc *BlockConcat) Combine(s *mx.Symbol, g ...*mx.Symbol) (*mx.Symbol, []*mx.Symbol, error) {
	var err error
	b := make([]*mx.Symbol, len(bc.blocks))
	for i, v := range bc.blocks {
		if b[i], _, err = v.Combine(s, g...); err != nil {
			return nil, nil, err
		}
	}
	return mx.Concat(b...), g, nil
}

func Concat(b ...Block) Block {
	return &BlockConcat{b}
}

type BlockStack struct {
	blocks []Block
	axis1  bool
}

func (bc *BlockStack) Combine(s *mx.Symbol, g ...*mx.Symbol) (*mx.Symbol, []*mx.Symbol, error) {
	var err error
	b := make([]*mx.Symbol, len(bc.blocks), len(bc.blocks))
	for i, v := range bc.blocks {
		if b[i], _, err = v.Combine(s, g...); err != nil {
			return nil, nil, err
		}
	}
	if bc.axis1 {
		s = mx.Stack1(b...)
	} else {
		s = mx.Stack(b...)
	}
	return s, g, nil
}

func TransStack(b ...Block) Block {
	return &BlockStack{b, true}
}

func Stack(b ...Block) Block {
	return &BlockStack{b, false}
}

type ResidualBlock struct {
	blocks []Block
}

func Residual(a ...Block) Block {
	return &ResidualBlock{a}
}

func (rcb *ResidualBlock) Combine(a *mx.Symbol, b ...*mx.Symbol) (*mx.Symbol, []*mx.Symbol, error) {
	var err error
	for _, n := range rcb.blocks {
		var c *mx.Symbol
		if c, b, err = n.Combine(a,b...); err != nil {
			return nil, nil, err
		}
		a = mx.Add(a,c)
	}
	return a, b, nil
}

