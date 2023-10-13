## Code block in details in a notice

{% capture exercise %}

<h3> Exercise </h3>
<p >With what you know so far, grab the scores josefine scored in the last games and compute the average amount of goals per game she scores</p>

{::options parse_block_html="true" /}

<details><summary markdown="span">Solution!</summary>

```python
scores = josefine_exampleperson["scores_last_games"]
total_scores = scores[0] + scores[1] + scores[2] + scores[3]
mean_scores = total_scores / 4
```
<p >obviously there are better solutions to this, for example the iteration over all scores can be done much better with a loop  
and the total number of score-values can be found using the len() function. A one-line solution could look like this:</p>

```python
mean_scores = sum(josefine_exampleperson["scores_last_games"]) / len(josefine_exampleperson["scores_last_games"])
```
</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--warning">
  {{ exercise | markdownify }}
</div>
