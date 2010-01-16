/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    MultipleEvaluation.java
 *    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
 *
 */
package mulan.evaluation;

import java.util.HashMap;
import mulan.evaluation.measure.Measure;

/**
 * Simple class that includes an array, whose elements are lists of evaluation
 * evaluations. Used to compute means and standard deviations of multiple
 * evaluations (e.g. cross-validation)
 * 
 * @author Grigorios Tsoumakas
 */
public class MultipleEvaluation {

    private Evaluation[] evaluations;
    private HashMap<String, Double> mean;
    private HashMap<String, Double> standardDeviation;

    public MultipleEvaluation(Evaluation[] evaluations) {
        this.evaluations = evaluations;
        calculateStatistics();
    }

    private void calculateStatistics() {
        int size = evaluations.length;
        HashMap<String, Double> sums = new HashMap<String, Double>();


        // calculate sums of measures
        for (int i = 0; i < evaluations.length; i++) {
            for (Measure m : evaluations[i].getMeasures()) {
                double value = Double.NaN;
                try {
                    value = m.getValue();
                } catch (Exception ex) {
                }
                if (sums.containsKey(m.getName())) {
                    sums.put(m.getName(), sums.get(m.getName()) + value);
                } else {
                    sums.put(m.getName(), value);
                }
            }
        }
        mean = new HashMap<String, Double>();
        for (String measureName : sums.keySet()) {
            mean.put(measureName, sums.get(measureName) / size);
        }

        // calculate sums of squared differences from mean
        sums = new HashMap<String, Double>();


        for (int i = 0; i < evaluations.length; i++) {
            for (Measure m : evaluations[i].getMeasures()) {
                double value = Double.NaN;
                try {
                    value = m.getValue();
                } catch (Exception ex) {
                }
                if (sums.containsKey(m.getName())) {
                    sums.put(m.getName(), sums.get(m.getName()) + Math.pow(value - mean.get(m.getName()), 2));
                } else {
                    sums.put(m.getName(), Math.pow(value - mean.get(m.getName()), 2));
                }
            }
        }
        standardDeviation = new HashMap<String, Double>();
        for (String measureName : sums.keySet()) {
            standardDeviation.put(measureName, Math.sqrt(sums.get(measureName) / size));
        }
    }

    @Override
    public String toString() {
        String description = "";
        for (Measure m : evaluations[0].getMeasures()) {
            String measureName = m.getName();
            description +=
                    (measureName + ": " +
                    mean.get(measureName) + "\u00B1" +
                    standardDeviation.get(measureName) + "\n");
        }

        return description;
    }

    public String toCSV() {
        String description = "";
        for (Measure m : evaluations[0].getMeasures()) {
            String measureName = m.getName();
            description +=
                    (mean.get(measureName) + ";" +
                    standardDeviation.get(measureName) + ";");
        }

        return description + "\n";
    }
}
