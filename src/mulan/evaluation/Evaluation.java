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
 *    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
 *
 */
package mulan.evaluation;

import java.util.List;
import mulan.evaluation.measure.Measure;

/**
 * Simple class that includes a list of evaluation measures returned from a
 * call to the static methods of {@link Evaluator} for evaluation purposes.
 * 
 * @see Evaluator
 * 
 * @author Jozef Vilcek
 * @author Grigorios Tsoumakas
 */
public class Evaluation {

    private List<Measure> measures;

    public Evaluation(List<Measure> measures) {
        this.measures = measures;
    }

    @Override
    public String toString() {
        String description = "";
        for (Measure m : measures) {
            description += m + "\n";
        }
        return description;
    }

    public String toCSV() {
        String description = "";
        for (Measure m : measures) {
            double value = Double.NaN;
            try {
                value = m.getValue();
            } catch (Exception ex) {
            }
            description += (value + ";");
        }
        return description + "\n";
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
