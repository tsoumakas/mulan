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
 *    DataLoadException.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.data;

import mulan.core.MulanRuntimeException;

/**
 * The exception is thrown to indicate an error while loading the data.
 * 
 * @author Jozef Vilcek
 */
public class DataLoadException extends MulanRuntimeException {

    private static final long serialVersionUID = 1102055196162204723L;

    /**
     * Creates a new instance of {@link DataLoadException} with detail mesage.
     * @param message the detail message
     */
    public DataLoadException(String message) {
        super(message);
    }

    /**
     * Creates a new instance of {@link DataLoadException} with detail message
     * and nested exception.
     *
     * @param message the detail message
     * @param cause the nested exception
     */
    public DataLoadException(String message, Throwable cause) {
        super(message, cause);
    }
}