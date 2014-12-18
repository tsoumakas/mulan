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
 *    Evaluation.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.evaluation;

import java.util.ArrayList;
import java.util.List;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.measure.MacroAverageMeasure;
import mulan.evaluation.measure.Measure;

/**
 * Simple class that includes a list of evaluation measures returned from a
 * call to the static methods of {@link Evaluator} for evaluation purposes.
 *
 * @see Evaluator
 *
 * @author Jozef Vilcek
 * @author Grigorios Tsoumakas
 * @version 2010.11.05
 */
public class Evaluation {

    private MultiLabelInstances data;
    private List<Measure> measures;

    /**
     * Creates a new evaluation object by deep copying the measure objects that are given as
     * parameters
     *
     * @param someMeasures calculated measures
     * @param data the evaluation data used for obtaining label names for outputting per label
     *            values of macro averaged measures
     * @throws Exception when creating Evaluation object fails
     */
    public Evaluation(List<Measure> someMeasures, MultiLabelInstances data) throws Exception {
        measures = new ArrayList<Measure>();
        for (Measure m : someMeasures) {
            Measure newMeasure = m.makeCopy();
            measures.add(newMeasure);
        }
        this.data = data;
    }

    /**
     * Returns a string with the results of the evaluation
     *
     * @return a string with the results of the evaluation
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (Measure m : measures) {
            sb.append(m);
            if (m instanceof MacroAverageMeasure) {
                sb.append("\n");
                for (int i = 0; i < data.getNumLabels(); i++) {
                    sb.append(data.getDataSet().attribute(data.getLabelIndices()[i]).name());
                    sb.append(": ");
                    sb.append(String.format("%.4f", ((MacroAverageMeasure) m).getValue(i)));
                    sb.append(" ");
                }
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    /**
     * Returns a CSV representation of the calculated measures
     *
     * @return the CSV representation of the calculated measures
     */
    public String toCSV() {
        StringBuilder sb = new StringBuilder();
        for (Measure m : measures) {
            double value = Double.NaN;
            try {
                value = m.getValue();
            } catch (Exception ex) {
            }
            sb.append(String.format("%.4f", value));
            sb.append(";");
        }
        return sb.toString();
    }

    /**
     * Returns the evaluation measures
     *
     * @return the evaluation measures
     */
    public List<Measure> getMeasures() {
        return measures;
    }
}
