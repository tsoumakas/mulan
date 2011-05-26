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
 *    FMeasure.java
 *    Copyright (C) 2009-2011 Aristotle University of Thessaloniki, Greece
 */
package mulan.evaluation.measure;

import mulan.core.MulanRuntimeException;

/**
 * Class for computing the FMeasure
 *
 * @author Grigorios Tsoumakas
 * @version 2010.12.31
 */
public class FMeasure {

    /**
     * The simple formula for the F-measure comes from van Rijsbergen's
     * effectiveness measure, which assumes that both precision and recall are
     * well-defined and different from zero
     * 
     * @param tp true positives
     * @param fp false positives
     * @param fn false negatives
     * @param beta 
     * @return the value of the f-measure
     */
    public static double compute(double tp, double fp, double fn, double beta) {
        if (tp + fp == 0) {
            throw new MulanRuntimeException("Zero positives predictions");
        }
        double precision = tp / (tp + fp);
        if (tp + fn == 0) {
            throw new MulanRuntimeException("Zero actual positives");
        }
        double recall = tp / (tp + fn);
        if (precision == 0 || recall == 0) {
            throw new MulanRuntimeException("The F-measure is undefined");
        } else {
            double beta2 = beta*beta;
            return ((beta2 + 1)*precision*recall)/(beta2*precision + recall);
        }
    }
}
