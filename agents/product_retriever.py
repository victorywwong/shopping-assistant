"""
LangChain tools for product retrieval.
These are bound to the orchestrator's LLM and executed by LangGraph's ToolNode.
"""

from langchain_core.tools import tool

from data.catalog import get_article
from data.embeddings import similarity_search


@tool
def search_products(
    query: str,
    colour_group_name: str | None = None,
    product_type_name: str | None = None,
    garment_group_name: str | None = None,
) -> str:
    """
    Search for women's clothing products by semantic query.
    Optionally filter by colour_group_name (e.g. 'Black', 'Pink'),
    product_type_name (e.g. 'Dress', 'Trousers'), or
    garment_group_name (e.g. 'Skirts', 'Jersey Fancy').
    Returns the top matching products with their article IDs.
    """
    filters = {
        k: v for k, v in {
            "colour_group_name": colour_group_name,
            "product_type_name": product_type_name,
            "garment_group_name": garment_group_name,
        }.items() if v
    }

    results = similarity_search(query, top_k=5, filters=filters or None)

    if not results:
        return (
            "No products found matching those criteria. "
            "Consider broadening the search by removing one of the filters."
        )

    lines = ["Here are the most relevant products:\n"]
    for r in results:
        lines.append(
            f"- **{r['prod_name']}** (article_id: {r['article_id']})\n"
            f"  Type: {r['product_type_name']} | Colour: {r['colour_group_name']}\n"
            f"  {r['detail_desc'] or ''}"
        )
    return "\n".join(lines)


@tool
def get_article_details(article_id: str) -> str:
    """
    Fetch full details for a specific product by its article_id.
    Use this when the customer asks for more information about a specific item.
    """
    article = get_article(article_id)
    if article is None:
        return f"No product found with article_id '{article_id}'."

    return (
        f"**{article['prod_name']}** (article_id: {article['article_id']})\n"
        f"Type: {article['product_type_name']} | Group: {article['product_group_name']}\n"
        f"Colour: {article['colour_group_name']} | Appearance: {article['graphical_appearance_name']}\n"
        f"Garment group: {article['garment_group_name']}\n"
        f"Description: {article['detail_desc']}"
    )
