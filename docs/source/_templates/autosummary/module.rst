{% extends "!autosummary/module.rst" %}


{% block classes %}
{% if classes %}
   .. rubric:: Classes

   {% for item in classes %}
   .. autoclass:: {{ fullname }}.{{ item }}
      :members:
      :inherited-members:
   {% endfor %}
{% endif %}
{% endblock %}

{% block functions %}
{% if functions %}
   .. rubric:: Functions

   {% for item in functions %}
   .. autofunction:: {{ fullname }}.{{ item }}
   {% endfor %}
{% endif %}
{% endblock %}