class Neuron
  attr_accessor :weights, :output, :error

  def initialize(num_of_inputs)
    @weights = []
    @output = nil
    @error = nil

    (num_of_inputs + 1).times do
      @weights << rand(-1.0..1.0)
    end
  end

  def activate(input)
    sum = 0

    input.each_index do |i|
      sum += input[i] * @weights[i]
    end

    sum += @weights.last * 1 # bias

    @output = activation_function(sum)
  end

  def activation_function(x)
    1 / (1 + Math.exp(-1 * x))
  end

  def train(input, answer, intensity = 0.1)
    guess = activate(input)
    @error = answer - guess

    self.update_weights(input, intensity)
  end

  def update_weights(input, intensity = 0.1)
    (0..@weights.length - 2).map do |i|
      @weights[i] += intensity * @error * input[i] 
    end

    @weights[@weights.length - 1] += intensity * @error * 1 # bias
  end
end
