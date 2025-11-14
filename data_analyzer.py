"""
Simplified Data Analyzer module
For detailed analysis, see data_analyzer.ipynb
"""

import logging
from typing import List, Dict, Any
from collections import Counter

logger = logging.getLogger(__name__)


class DataAnalyzer:
    """Simple data analyzer for API endpoint"""
    
    async def analyze(self, member_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform basic analysis of member data
        
        Args:
            member_data: List of all member messages
        
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Analyzing {len(member_data)} messages")
        
        # Count unique members
        members = set()
        member_message_count = Counter()
        message_lengths = []
        
        for msg in member_data:
            member_name = msg.get("member_name", "Unknown")
            members.add(member_name)
            member_message_count[member_name] += 1
            message_lengths.append(len(msg.get("content", "")))
        
        # Calculate basic statistics
        total_messages = len(member_data)
        unique_members = len(members)
        avg_message_length = sum(message_lengths) / len(message_lengths) if message_lengths else 0
        
        # Find most active member
        most_active = member_message_count.most_common(1)[0] if member_message_count else ("None", 0)
        
        results = {
            "summary": f"Analyzed {total_messages} messages from {unique_members} unique members.",
            "statistics": {
                "total_messages": total_messages,
                "unique_members": unique_members,
                "avg_messages_per_member": round(total_messages / unique_members, 2) if unique_members else 0,
                "avg_message_length": round(avg_message_length, 2),
                "most_active_member": {
                    "name": most_active[0],
                    "message_count": most_active[1]
                }
            },
            "message": "For detailed analysis with visualizations, run the data_analyzer.ipynb notebook"
        }
        
        logger.info("Analysis complete")
        return results
