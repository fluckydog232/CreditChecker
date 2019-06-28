module Assignment3

class BinomialModel
  def func dataset, w
    mu = w["mu"]
    dataset.inject(0.0) do |u,row|
      x = row["label"]
      u -= Math.log((mu ** x) * ((1 - mu) ** (1 - x)))
    end
  end

  def grad dataset, w
    mu = w["mu"]
    g = Hash.new {|h,k| h[k] = 0.0}
    dataset.each do |row|
      x = row["label"]
      g["mu"] -= (x / mu) - (1 - x) / (1 - mu)
    end
    return g
  end

  ## Adjusts the parameter to be within the allowable range
  def adjust w
    w["mu"] = [[0.001, w["mu"]].max, 0.999].min
  end
end

class NaiveBayesModel
  def func dataset, w
    -dataset.inject(0.0) do |u, row|
      cls = row["label"].to_f > 0 ? "pos" : "neg"
      p = cls == "pos" ? 1.0 : 0.0
      u += Math.log((w["pos_bias"] ** p) * ((1 - w["pos_bias"]) ** (1 - p)))
      n = cls == "neg" ? 1.0 : 0.0
      u += Math.log((w["neg_bias"] ** n) * ((1 - w["neg_bias"]) ** (1 - n)))

      u += row["features"].keys.inject(Math.log(w["#{cls}_bias"])) do |u, fname|
        u += Math.log(w["#{cls}_#{fname}"]) * row["features"][fname]
      end
    end
  end

  def grad dataset, w
    g = Hash.new {|h,k| h[k] = 0.0}
    dataset.each do |row|
      cls = row["label"].to_f > 0 ? "pos" : "neg"
      p = cls == "pos" ? 1.0 : 0.0
      g["pos_bias"] -= (p / w["pos_bias"]) - (1 - p) / (1 - w["pos_bias"])

      n = cls == "neg" ? 1.0 : 0.0
      g["neg_bias"] -= (n / w["neg_bias"]) - (1 - n) / (1 - w["neg_bias"])


      row["features"].each_key do |fname|
        g["#{cls}_#{fname}"] -= row["features"][fname] / w["#{cls}_#{fname}"]
      end
    end
    return g
  end

  def predict row, w
    s = Hash.new {|h,k| h[k] = 0.0}
    puts row["features"]

    %w(pos neg).each do |j|
      row["features"].keys.each do |k|
        p_jk = j + "_" + k
        s[k] += row["features"][j] * Math.log(w[p_jk])
      end
    end
    score = s["pos"] - s["neg"]
    return score
  end
  # def predict row, w
  #   scores = Hash.new
  #
  #   %w(pos neg).each do |cls|
  #     scores[cls] = row["features"].keys.inject(Math.log(w["#{cls}_bias"])) do |u, fname|
  #       u += Math.log(w["#{cls}_#{fname}"]) * row["features"][fname]
  #     end
  #   end
  #   score = scores["pos"] - scores["neg"]
  # end
  def adjust w
    w.each_key do |fname|
      w[fname] = [[0.001, w[fname]].max, 0.999].min
    end
  end
end

def coin_dataset(n)
  header = %w(x)
  p = 0.7743
  dataset = []
  n.times do
    outcome = rand < p ? 1.0 : 0.0
    dataset << {"features" => {"bias" => 1.0}, "label" => outcome}
  end
  return [header, dataset]
end

def plot x, y
  Daru::DataFrame.new({x: x, y: y}).plot(type: :line, x: :x, y: :y) do |plot, diagram|
    plot.x_label "X"
    plot.y_label "Y"
  end
end

end

# other toolkit
#calculate the deviation for features
# def std(x)
#   m = mean x
#   l = (x.length-1).to_f
#   u = x.reduce(0.0) do |u, p|
#       u += p.nil? ? 0 : ((p-m)**2 / l)
#   end
# return Math.sqrt u
# end

def norm w
    0.5 * Math.sqrt(w.keys.inject(0.0) {|u,k| u += w[k] ** 2.0})
end

# dot mulitply
def dot row, w
  f = row["features"]
  return f.keys.inject(0.0) {|u, k| u += f[k] * w[k]}
end
