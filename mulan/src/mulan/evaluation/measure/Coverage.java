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
 *    Coverage.java
 *    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
 *
 */
package mulan.evaluation.measure;


import mulan.classifier.MultiLabelOutput;

/**
 * Implementation of the coverage measure.
 * 
 * @author Grigorios Tsoumakas
 */
public class Coverage extends ExampleBasedMeasure {

    public String getName() {
        return "Coverage";
    }

    public double updateInternal(MultiLabelOutput output, boolean[] trueLabels) {

        int howDeep = 0;
        int numLabels = trueLabels.length;
        int[] ranks = output.getRanking();
        for (int rank = numLabels; rank >= 1; rank--) {
            int indexOfRank;
            for (indexOfRank = 0; indexOfRank < numLabels; indexOfRank++) {
                if (ranks[indexOfRank] == rank) {
                    break;
                }
            }
            if (trueLabels[indexOfRank]) {
                howDeep = rank - 1;
                break;
            }
        }
        
        sum += howDeep;
        count++;

        return howDeep;
    }

    @Override
    public double getIdealValue() {
        return 1;
    }
}
