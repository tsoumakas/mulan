# Multi-label datasets

The following multi-label datasets are properly formatted for use with Mulan. We initially provide a table with dataset statistics, followed by the actual files and sources. 

## Statistics

<table border="0" cellpadding="2" cellspacing="1" width="200" style="font-size:0.9em">
          <tbody>
            <tr style="background-color:#D4F8BC">
              <td nowrap="nowrap"> </td>
              <td align="center" nowrap="nowrap"> </td>
              <td nowrap="nowrap"> </td>
              <td colspan="2" align="center" nowrap="nowrap"> <strong>attributes</strong></td>
              <td align="center" nowrap="nowrap"></td>
              <td nowrap="nowrap"><strong> </strong></td>
              <td nowrap="nowrap"><strong> </strong></td>
              <td nowrap="nowrap"><strong> </strong></td>
            </tr>
            <tr style="background-color:#D4F8BC">
              <td nowrap="nowrap"><strong>name</strong></td>
              <td align="center" nowrap="nowrap"><strong>domain</strong></td>
              <td align="center" nowrap="nowrap" ><strong>instances</strong></td>
              <td align="center" nowrap="nowrap" ><strong>nominal</strong></td>
              <td align="center" nowrap="nowrap" ><strong>numeric</strong></td>
              <td align="center" nowrap="nowrap" ><strong>labels</strong></td>
              <td align="center" nowrap="nowrap" ><strong>cardinality</strong></td>
              <td align="center" nowrap="nowrap" ><strong>density</strong></td>
              <td align="center" nowrap="nowrap" ><strong>distinct </strong></td>
            </tr>
            <tr style="background-color:#f7f7f7">
              <td nowrap="nowrap">bibtex</td>
              <td align="center" nowrap="nowrap">text</td>
              <td align="right" nowrap="nowrap">7395</td>
              <td align="right" nowrap="nowrap">1836</td>
              <td align="right" nowrap="nowrap">0</td>
              <td align="right" nowrap="nowrap">159</td>
              <td align="right" nowrap="nowrap">2.402</td>
              <td align="right" nowrap="nowrap">0.015</td>
              <td align="right" nowrap="nowrap">2856</td>
            </tr>
            <tr style="background-color:#f0f0f0">
              <td nowrap="nowrap">birds</td>
              <td align="center" nowrap="nowrap">audio</td>
              <td align="right" nowrap="nowrap">645</td>
              <td align="right" nowrap="nowrap">2</td>
              <td align="right" nowrap="nowrap">258</td>
              <td align="right" nowrap="nowrap">19</td>
              <td align="right" nowrap="nowrap">1.014</td>
              <td align="right" nowrap="nowrap">0.053</td>
              <td align="right" nowrap="nowrap">133</td>
            </tr>
            <tr style="background-color:#f7f7f7">
              <td nowrap="nowrap">bookmarks</td>
              <td align="center" nowrap="nowrap">text</td>
              <td align="right" nowrap="nowrap">87856</td>
              <td align="right" nowrap="nowrap">2150</td>
              <td align="right" nowrap="nowrap">0</td>
              <td align="right" nowrap="nowrap">208</td>
              <td align="right" nowrap="nowrap">2.028</td>
              <td align="right" nowrap="nowrap">0.010</td>
              <td align="right" nowrap="nowrap">18716</td>
            </tr>
            <tr style="background-color:#f0f0f0">
              <td nowrap="nowrap">CAL500 </td>
              <td align="center" nowrap="nowrap">music</td>
              <td align="right" nowrap="nowrap">502</td>
              <td align="right" nowrap="nowrap">0</td>
              <td align="right" nowrap="nowrap">68</td>
              <td align="right" nowrap="nowrap">174</td>
              <td align="right" nowrap="nowrap">26.044</td>
              <td align="right" nowrap="nowrap">0.150</td>
              <td align="right" nowrap="nowrap">502</td>
            </tr>
            <tr style="background-color:#f7f7f7">
              <td nowrap="nowrap">corel5k </td>
              <td align="center" nowrap="nowrap">images</td>
              <td align="right" nowrap="nowrap">5000</td>
              <td align="right" nowrap="nowrap">499</td>
              <td align="right" nowrap="nowrap">0</td>
              <td align="right" nowrap="nowrap">374</td>
              <td align="right" nowrap="nowrap">3.522</td>
              <td align="right" nowrap="nowrap">0.009</td>
              <td align="right" nowrap="nowrap">3175</td>
            </tr>
            <tr style="background-color:#f0f0f0">
              <td nowrap="nowrap">corel16k (10 samples) </td>
              <td align="center" nowrap="nowrap">images</td>
              <td align="right" nowrap="nowrap">13811&#177;87 </td>
              <td align="right" nowrap="nowrap">500</td>
              <td align="right" nowrap="nowrap">0</td>
              <td align="right" nowrap="nowrap">161&#177;9</td>
              <td align="right" nowrap="nowrap">2.867&#177;0.033</td>
              <td align="right" nowrap="nowrap">0.018&#177;0.001</td>
              <td align="right" nowrap="nowrap">4937&#177;158</td>
            </tr>
            <tr style="background-color:#f7f7f7">
              <td nowrap="nowrap">delicious</td>
              <td align="center" nowrap="nowrap">text (web) </td>
              <td align="right" nowrap="nowrap">16105</td>
              <td align="right" nowrap="nowrap">500</td>
              <td align="right" nowrap="nowrap">0</td>
              <td align="right" nowrap="nowrap">983</td>
              <td align="right" nowrap="nowrap">19.020</td>
              <td align="right" nowrap="nowrap">0.019</td>
              <td align="right" nowrap="nowrap">15806</td>
            </tr>
            <tr style="background-color:#f0f0f0">
              <td nowrap="nowrap">emotions</td>
              <td align="center" nowrap="nowrap">music</td>
              <td align="right" nowrap="nowrap">593</td>
              <td align="right" nowrap="nowrap">0</td>
              <td align="right" nowrap="nowrap">72</td>
              <td align="right" nowrap="nowrap">6</td>
              <td align="right" nowrap="nowrap">1.869</td>
              <td align="right" nowrap="nowrap">0.311</td>
              <td align="right" nowrap="nowrap">27</td>
            </tr>
            <tr style="background-color:#f7f7f7">
              <td nowrap="nowrap">enron</td>
              <td align="center" nowrap="nowrap">text</td>
              <td align="right" nowrap="nowrap">1702</td>
              <td align="right" nowrap="nowrap">1001</td>
              <td align="right" nowrap="nowrap">0</td>
              <td align="right" nowrap="nowrap">53</td>
              <td align="right" nowrap="nowrap">3.378</td>
              <td align="right" nowrap="nowrap">0.064</td>
              <td align="right" nowrap="nowrap">753</td>
            </tr>
            <tr style="background-color:#f0f0f0">
              <td nowrap="nowrap">EUR-Lex (directory codes) </td>
              <td align="center" nowrap="nowrap">text</td>
              <td align="right" nowrap="nowrap">19348</td>
              <td align="right" nowrap="nowrap">0</td>
              <td align="right" nowrap="nowrap">5000</td>
              <td align="right" nowrap="nowrap">412</td>
              <td align="right" nowrap="nowrap">1.292</td>
              <td align="right" nowrap="nowrap">0.003</td>
              <td align="right" nowrap="nowrap">1615</td>
            </tr>
            <tr style="background-color:#f7f7f7">
              <td nowrap="nowrap">EUR-Lex (subject matters) </td>
              <td align="center" nowrap="nowrap">text</td>
              <td align="right" nowrap="nowrap">19348</td>
              <td align="right" nowrap="nowrap">0</td>
              <td align="right" nowrap="nowrap">5000</td>
              <td align="right" nowrap="nowrap">201</td>
              <td align="right" nowrap="nowrap">2.213</td>
              <td align="right" nowrap="nowrap">0.011</td>
              <td align="right" nowrap="nowrap">2504</td>
            </tr>
            <tr style="background-color:#f0f0f0">
              <td nowrap="nowrap">EUR-Lex (eurovoc descriptors) </td>
              <td align="center" nowrap="nowrap">text</td>
              <td align="right" nowrap="nowrap">19348</td>
              <td align="right" nowrap="nowrap">0</td>
              <td align="right" nowrap="nowrap">5000</td>
              <td align="right" nowrap="nowrap">3993</td>
              <td align="right" nowrap="nowrap">5.310</td>
              <td align="right" nowrap="nowrap">0.001</td>
              <td align="right" nowrap="nowrap">16467</td>
            </tr>
            <tr style="background-color:#f0f0f0">
              <td nowrap="nowrap">flags</td>
              <td align="center" nowrap="nowrap">images (toy)</td>
              <td align="right" nowrap="nowrap">194</td>
              <td align="right" nowrap="nowrap">9</td>
              <td align="right" nowrap="nowrap">10</td>
              <td align="right" nowrap="nowrap">7</td>
              <td align="right" nowrap="nowrap">3.392</td>
              <td align="right" nowrap="nowrap">0.485</td>
              <td align="right" nowrap="nowrap">54</td>
            </tr>
            <tr style="background-color:#f7f7f7">
              <td nowrap="nowrap">foodtruck</td>
              <td align="center" nowrap="nowrap">other</td>
              <td align="right" nowrap="nowrap">407</td>
              <td align="right" nowrap="nowrap">4</td>
              <td align="right" nowrap="nowrap">17</td>
              <td align="right" nowrap="nowrap">12</td>
              <td align="right" nowrap="nowrap">2.289</td>
              <td align="right" nowrap="nowrap">0.191</td>
              <td align="right" nowrap="nowrap">116</td>
            </tr>
            <tr style="background-color:#f0f0f0">
              <td nowrap="nowrap" style="height: 23px">genbase</td>
              <td align="center" nowrap="nowrap" style="height: 23px">biology</td>
              <td align="right" nowrap="nowrap" style="height: 23px">662</td>
              <td align="right" nowrap="nowrap" style="height: 23px">1186</td>
              <td align="right" nowrap="nowrap" style="height: 23px">0</td>
              <td align="right" nowrap="nowrap" style="height: 23px">27</td>
              <td align="right" nowrap="nowrap" style="height: 23px">1.252</td>
              <td align="right" nowrap="nowrap" style="height: 23px">0.046</td>
              <td align="right" nowrap="nowrap" style="height: 23px">32</td>
            </tr>
            <tr style="background-color:#f7f7f7">
              <td nowrap="nowrap">mediamill</td>
              <td align="center" nowrap="nowrap">video</td>
              <td align="right" nowrap="nowrap">43907</td>
              <td align="right" nowrap="nowrap">0</td>
              <td align="right" nowrap="nowrap">120</td>
              <td align="right" nowrap="nowrap">101</td>
              <td align="right" nowrap="nowrap">4.376</td>
              <td align="right" nowrap="nowrap">0.043</td>
              <td align="right" nowrap="nowrap">6555</td>
            </tr>
            <tr style="background-color:#f0f0f0">
              <td nowrap="nowrap">medical</td>
              <td align="center" nowrap="nowrap">text</td>
              <td align="right" nowrap="nowrap">978</td>
              <td align="right" nowrap="nowrap">1449</td>
              <td align="right" nowrap="nowrap">0</td>
              <td align="right" nowrap="nowrap">45</td>
              <td align="right" nowrap="nowrap">1.245</td>
              <td align="right" nowrap="nowrap">0.028</td>
              <td align="right" nowrap="nowrap">94</td>
            </tr>
            <tr style="background-color:#f7f7f7">
              <td nowrap="nowrap">NUS-WIDE</td>
              <td align="center" nowrap="nowrap">images</td>
              <td align="right" nowrap="nowrap">269648</td>
              <td align="right" nowrap="nowrap">0</td>
              <td align="right" nowrap="nowrap">128/500</td>
              <td align="right" nowrap="nowrap">81</td>
              <td align="right" nowrap="nowrap">1.869</td>
              <td align="right" nowrap="nowrap">0.023</td>
              <td align="right" nowrap="nowrap">18430</td>
            </tr>
            <tr style="background-color:#f0f0f0">
              <td nowrap="nowrap">rcv1v2 (subset1)</td>
              <td align="center" nowrap="nowrap">text</td>
              <td align="right" nowrap="nowrap">6000</td>
              <td align="right" nowrap="nowrap">0</td>
              <td align="right" nowrap="nowrap">47236</td>
              <td align="right" nowrap="nowrap">101</td>
              <td align="right" nowrap="nowrap">2.880</td>
              <td align="right" nowrap="nowrap">0.029</td>
              <td align="right" nowrap="nowrap">1028</td>
            </tr>
            <tr style="background-color:#f7f7f7">
              <td nowrap="nowrap">rcv1v2 (subset2)</td>
              <td align="center" nowrap="nowrap">text</td>
              <td align="right" nowrap="nowrap">6000</td>
              <td align="right" nowrap="nowrap">0</td>
              <td align="right" nowrap="nowrap">47236</td>
              <td align="right" nowrap="nowrap">101</td>
              <td align="right" nowrap="nowrap">2.634</td>
              <td align="right" nowrap="nowrap">0.026</td>
              <td align="right" nowrap="nowrap">954</td>
            </tr>
            <tr style="background-color:#f0f0f0">
              <td nowrap="nowrap">rcv1v2 (subset3)</td>
              <td align="center" nowrap="nowrap">text</td>
              <td align="right" nowrap="nowrap">6000</td>
              <td align="right" nowrap="nowrap">0</td>
              <td align="right" nowrap="nowrap">47236</td>
              <td align="right" nowrap="nowrap">101</td>
              <td align="right" nowrap="nowrap">2.614</td>
              <td align="right" nowrap="nowrap">0.026</td>
              <td align="right" nowrap="nowrap">939</td>
            </tr>
            <tr style="background-color:#f7f7f7">
              <td nowrap="nowrap">rcv1v2 (subset4)</td>
              <td align="center" nowrap="nowrap">text</td>
              <td align="right" nowrap="nowrap">6000</td>
              <td align="right" nowrap="nowrap">0</td>
              <td align="right" nowrap="nowrap">47229</td>
              <td align="right" nowrap="nowrap">101</td>
              <td align="right" nowrap="nowrap">2.484</td>
              <td align="right" nowrap="nowrap">0.025</td>
              <td align="right" nowrap="nowrap">816</td>
            </tr>
            <tr style="background-color:#f0f0f0">
              <td nowrap="nowrap">rcv1v2 (subset5)</td>
              <td align="center" nowrap="nowrap">text</td>
              <td align="right" nowrap="nowrap">6000</td>
              <td align="right" nowrap="nowrap">0</td>
              <td align="right" nowrap="nowrap">47235</td>
              <td align="right" nowrap="nowrap">101</td>
              <td align="right" nowrap="nowrap">2.642</td>
              <td align="right" nowrap="nowrap">0.026</td>
              <td align="right" nowrap="nowrap">946</td>
            </tr>
            <tr style="background-color:#f7f7f7">
              <td nowrap="nowrap">scene</td>
              <td align="center" nowrap="nowrap">image</td>
              <td align="right" nowrap="nowrap">2407</td>
              <td align="right" nowrap="nowrap">0</td>
              <td align="right" nowrap="nowrap">294</td>
              <td align="right" nowrap="nowrap">6</td>
              <td align="right" nowrap="nowrap">1.074</td>
              <td align="right" nowrap="nowrap">0.179</td>
              <td align="right" nowrap="nowrap">15</td>
            </tr>
            <tr style="background-color:#f0f0f0">
              <td nowrap="nowrap">tmc2007</td>
              <td align="center" nowrap="nowrap">text</td>
              <td align="right" nowrap="nowrap">28596</td>
              <td align="right" nowrap="nowrap">49060<br />                  </td>
              <td align="right" nowrap="nowrap">0</td>
              <td align="right" nowrap="nowrap">22</td>
              <td align="right" nowrap="nowrap">2.158</td>
              <td align="right" nowrap="nowrap">0.098</td>
              <td align="right" nowrap="nowrap">1341</td>
            </tr>
            <tr style="background-color:#f7f7f7">
              <td nowrap="nowrap">yahoo</td>
              <td align="center" nowrap="nowrap">text</td>
              <td align="right" nowrap="nowrap">5423±1259</td>
              <td align="right" nowrap="nowrap">0</td>
              <td align="right" nowrap="nowrap">32786±7990</td>
              <td align="right" nowrap="nowrap">31±6</td>
              <td align="right" nowrap="nowrap">1.481±0.154</td>
              <td align="right" nowrap="nowrap">0.051±0.012</td>
              <td align="right" nowrap="nowrap">321±139</td>
            </tr>
            <tr style="background-color:#f0f0f0">
              <td nowrap="nowrap">yeast</td>
              <td align="center" nowrap="nowrap">biology</td>
              <td align="right" nowrap="nowrap">2417</td>
              <td align="right" nowrap="nowrap">0</td>
              <td align="right" nowrap="nowrap">103</td>
              <td align="right" nowrap="nowrap">14</td>
              <td align="right" nowrap="nowrap">4.237</td>
              <td align="right" nowrap="nowrap">0.303</td>
              <td align="right" nowrap="nowrap">198</td>
            </tr>
          </tbody>
        </table>


