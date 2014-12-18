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
 *    LabelsPair.java
 */
package mulan.data;

import java.io.Serializable;

/**
 * Class for handling label pairs along with their 'dependence' scores. A pair of labels is represented as an array of two int values.
 * The natural order of label pairs is according to natural order of their dependence scores.
 *
 * @author Lena Chekina (lenat@bgu.ac.il)
 * @version 30.11.2010
 */
public class LabelsPair implements Comparable, Serializable {
    /** a pair of labels*/
    int[] pair;
    /** dependence score of the labels pair*/
    Double score;

    /**
     * Initialize a labels pair using an array of two int values.
     * @param aPair - a pair of labels (i.e. an array of length 2)
     * @param aScore - a dependence score
     */
    public LabelsPair(int[] aPair, double aScore) {
        if (aPair.length != 2) {
            throw new IllegalArgumentException("aPair should be of length 2!");
        }
        pair = aPair;
        score = aScore;
    }

    /**
     * 
     * @return The pair of labels
     */
    public int[] getPair() {
        return pair;
    }

    /**
     * 
     * @param pair a pair of labels
     */
    public void setPair(int[] pair) {
        this.pair = pair;
    }

    /**
     * 
     * @return The dependence score of the labels pair
     */
    public Double getScore() {
        return score;
    }

    /**
     * 
     * @param score the dependence score of the labels pair
     */
    public void setScore(Double score) {
        this.score = score;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        LabelsPair pair = (LabelsPair) o;
        return !(score != null ? !score.equals(pair.score) : pair.score != null);
    }

    public int compareTo(Object otherPair) {
        if( otherPair == null ) {
            throw new NullPointerException();
        }
        if( !( otherPair instanceof LabelsPair)) {
            throw new ClassCastException("Invalid object");
        }
        Double value = ( (LabelsPair) otherPair ).getScore();
        if(  this.getScore() > value )
            return 1;
        else if ( this.getScore() < value )
            return -1;
        else
            return 0;
    }

    @Override
    public int hashCode() {
        return score != null ? score.hashCode() : 0;
    }

    /**
     * @return a string representation of a labels pair in the form: "labels pair: [index1, index2]; dependence score; value"
     */
    @Override
    public String toString() {
        return  "labels pair: [" + pair[0] + ", " + pair[1] + "]; " +
                " dependence score; " + score + '\n';
    }
}