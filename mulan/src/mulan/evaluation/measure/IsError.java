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
 *    IsError.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.evaluation.measure;

import java.util.ArrayList;
import java.util.List;

/**
 * Implementation of is-error measure. The measure just indicates if the ranking is perfect 
 * or not. Speaking in terms of error set, the measure returns zero if cardinality of the
 * error-set is zero, and returns one if cardinality of the error set is greather than zero.
 * <br></br> 
 * The error set is defined as: set, composed of all possible label pairs, where one is relevant and 
 * the other is not, and which satisfies condition that relevant label is ranked lower than irrelevant.
 * 
 * @author Jozef Vilcek
 */
public class IsError extends RankingMeasureBase {

    public String getName() {
        return "Is-Error";
    }

    public double updateInternal2(int[] ranking, boolean[] trueLabels) {

        double isError = 0;
        int numLabels = trueLabels.length;
        List<Integer> relevant = new ArrayList<Integer>();
        List<Integer> irrelevant = new ArrayList<Integer>();
        for (int index = 0; index < numLabels; index++) {
            if (trueLabels[index]) {
                relevant.add(index);
            } else {
                irrelevant.add(index);
            }
        }

        boolean terminate = false;
        for (int rLabel : relevant) {
            for (int irLabel : irrelevant) {
                if (ranking[rLabel] > ranking[irLabel]) {
                    isError = 1;
                    terminate = true;
                    break;
                }
            }
            if (terminate) {
                break;
            }
        }

        sum += isError;
        count++;
        return isError;
    }

    @Override
    public double getIdealValue() {
        return 0;
    }
}
