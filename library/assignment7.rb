
require 'json'

module Assignment7
  def pick_feature_all db, table, feature, with_id = false
    query = "select " + feature + ", Target label from " + table + " where " + feature + " IS NOT NULL and " + feature + " != '' order by " + feature + " asc"
    query = "select SK_ID_CURR, " +  feature + ", Target label from " + table + " order by " + feature + " asc" if with_id
    data = db.execute(query)
    return data
  end

  # turns data to data_class
  def parse_class x
    s = Hash.new {|h,k| h[k] = 0}
    x.each do |p|
      s[p["label"]] += 1
    end
    return s
  end

  # Binary split on categorical features
  def parse_category x, col
    s = Hash.new {|h,k| h[k] = 0}
    x.each do |p|
      s[p[col]] += 1
    end
    return s
  end

  def p_log_p(x)
    return ( x == 0 ? 0 : x * Math.log(x,2) )
  end

  def entropy p
    total = total_instance(p)
    return -p.values.reduce(0.0) do |u, pi|
      u += p_log_p(pi/total)
    end
  end

  # x is the entropy of total instances, n is the number of entropy
  def information_gain x, n, splits
  #     puts splits
    return splits.reduce(x) do |u, s|
  #     puts u
  #     abort("test here")
      u -= total_instance(s)/n*entropy(s)
    end
  end

  def total_instance s
    return s.values.reduce(0.0) do |t, p|
      t += p
    end
  end

  def best_ig data, feature, type
    max_ig = 0
    best_v = 0
    c = parse_class data
    e = entropy(c)
    n = data.length
    left = Hash.new {|h,k| h[k] = 0}
    right = parse_class data

    groups = data.group_by{|h| h[feature]}

    if type == "TEXT"
      # categorical features
      groups.each do |p|
        p[1].each do |pi|
          left[pi["label"]] += 1
          right[pi["label"]] -= 1
        end

        ig = information_gain(e, n, [left, right])

        if ig > max_ig
          max_ig = ig
          best_v = p[0]
        end
        left = Hash.new {|h,k| h[k] = 0}
        right = parse_class data

      end
    else
      # numericals goes here
      groups.each do |p|

          p[1].each do |pi|
            left[pi["label"]] += 1
            right[pi["label"]] -= 1
          end
          ig = information_gain(e, n, [left, right])

          if ig > max_ig
            max_ig = ig
            best_v = p[0]
          end
        end
    end
    return [max_ig, best_v]
  end


  def fetch_data db, features, limit = false
  table = "application_train"
  query = "select " + features.join(",") + ", Target label from " + table
  query += " limit 10000" if limit
  data = []
  db.execute(query).each do |r|
    row = Hash.new
    row["features"] = Hash.new
    features.each do |f|
      row["features"][f] = r[f]
    end
    row["label"] = r["label"]
    data << row
  end
  return data
end

end
