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
</details>

{::options parse_block_html="false" /}

{% endcapture %}

<div class="notice--primary">
  {{ exercise | markdownify }}
</div>
