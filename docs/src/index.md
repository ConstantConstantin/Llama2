```@meta
CurrentModule = Llama2
```

# Llama2.jl

## What is Llama2?

LLama2 is a family of pre-trained LLMs by Meta AI. More information can be found at: https://www.llama.com/

## What is Llama2.jl?

[Llama2.jl](https://github.com/ConstantConstantin/Llama2.jl) can inference a given model from within `julia`. For this cause you will have to provide your own model checkpoint. This project follows the procedure outlined by the `run.c` file from [llama2.c](https://github.com/karpathy/llama2.c).

## Getting started

Start julia, activate a desired environment and add the package:

```julia
(@v1.11) pkg> activate .

(myLlama2) pkg> add https://github.com/ConstantConstantin/Llama2.jl
```

In every subsequent session it can be loaded via:

```julia
julia> using Llama2
```

## Example Usage

```julia
julia> print(talktollm("/PATH/TO/YOUR/MODEL.bin", "In a small village "))
In a small village house, there was a man named Tom. Tom was kind and would always shine his in front of the town. People from the village would come to look at Tom and feel happy.
One day, a little girl named Lily came to Tom. She did not have a passport. Tom saw Lily and said, "Why don't you have a passport, Lily? Hop in and pass me a little in our country!" Lily smiled and said, "Yes, I feel comfortable when I am in my own nation!"
Lily put on her sunglasses and they became good friends. The town was filled with happy puppies who shared their sunglasses with everyone. The people in the town knew that being kind and working together made everything better.
```

```@index
```
