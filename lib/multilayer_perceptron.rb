require_relative 'neuron'

class MultilayerPerceptron
  attr_accessor :layers

  def initialize(num_of_ins, hidden_layers)
    @layers = []

    hidden_layers.each_with_index do |n, i|
      ins_num = (i == 0 ? num_of_ins : @layers[i - 1].count)
      @layers[i] = []
      n.times do
        @layers[i] << Neuron.new(ins_num)
      end
    end
  end

  def feed(input)
    @layers.each_with_index do |layer, i|
      layer.each do |neuron|
        if i == 0
          neuron.activate(input)
        else
          input = @layers[i - 1].map { |n| n.output }
          neuron.activate(input)
        end
      end
    end

    return @layers.last.map { |n| n.output }
  end

  def train(input, answer, intensity = 0.1)
    self.feed(input)

    r_layers = @layers.reverse

    r_layers.each_with_index do |layer, i|

      # propagate error
      if i == 0
        layer.each_with_index do |n, n_i|
          n.error = n.output * (1 - n.output) * (answer[n_i] - n.output)
        end 
      else
        layer.each_with_index do |n, n_i|
          error = 0
          r_layers[i - 1].each do |o_n|
            error += o_n.error * o_n.weights[n_i]
          end
          n.error = n.output * (1 - n.output) * error
        end
      end

      # update weights
      if i == (r_layers.length - 1)
        layer.each do |n|
          n.update_weights(input, intensity)
        end
      elsif i == 0
        prev_input = @layers.length < 2 ? input : @layers[-2].map { |n| n.output }
        layer.each do |n|
          n.update_weights(prev_input, intensity)
        end
      else
        prev_input = r_layers[i + 1].map { |n| n.output }
        layer.each do |n|
          n.update_weights(prev_input, intensity)
        end
      end
    end
  end
end
